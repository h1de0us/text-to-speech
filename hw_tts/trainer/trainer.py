import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torchaudio

from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker

from generate_audio import synthesis, get_data
import os
import audio
import waveglow
from utils import get_WaveGlow
from hw_tts.utils import ROOT_PATH


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.metrics = MetricTracker(
            "loss", 
            "grad_norm",
            "mel_loss", 
            "duration_predictor_loss", 
            "pitch_predictor_losss",
            "energy_predictor_loss", 
            writer=self.writer
        )


    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        tqdm_bar = tqdm(range(self.len_epoch), desc='train')
        for li, lb in enumerate(self.train_dataloader):
            for batch_idx, batch in enumerate(lb):
                self.model.train()
                tqdm_bar.update(1)
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.metrics.update("grad_norm", self.get_grad_norm())
                batch_idx_ = batch_idx + li * batch["batch_expand_size"][0]
                if batch_idx_ % self.log_step == 0:
                    self.writer.set_step(
                        (epoch - 1) * self.len_epoch * self.batch_expand_size + batch_idx_
                    )
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_scalars(self.metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.metrics.result()
                    self.metrics.reset()
            if li >= self.len_epoch - 1:
                break
        log = last_train_metrics
        self._evaluation_epoch(epoch)
        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device, is_train)
        if is_train:
            self.optimizer.zero_grad()

        character = batch["text"].long().to(self.device)
        mel_target = batch["mel_target"].float().to(self.device)
        duration = batch["duration"].int().to(self.device)
        mel_pos = batch["mel_pos"].long().to(self.device)
        src_pos = batch["src_pos"].long().to(self.device)
        max_mel_len = batch["mel_max_len"]
        output, duration_predictor_output = self.model(character, 
                                                       src_pos, 
                                                       mel_pos, 
                                                       max_mel_len,
                                                       length_target=duration,)
        batch["mel_output"] = output
        batch["duration_predictor_output"] = duration_predictor_output

        mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss = self.criterion(**batch)
        total_loss = mel_loss + duration_predictor_loss
        batch["mel_loss"] = mel_loss
        metrics.update("mel_loss", batch["mel_loss"].item())
        batch["duration_predictor_loss"] = duration_predictor_loss
        metrics.update("duration_predictor_loss", batch["duration_predictor_loss"].item())

        if energy_predictor_loss is not None:
            batch["energy_predictor_loss"] = energy_predictor_loss
            metrics.update("energy_predictor_loss", batch["energy_predictor_loss"].item())
            total_loss += energy_predictor_loss
        if pitch_predictor_loss is not None:
            batch["pitch_predictor_loss"] = pitch_predictor_loss
            metrics.update("pitch_predictor_loss", batch["pitch_predictor_loss"].item())
            total_loss += pitch_predictor_loss
        batch["loss"] = total_loss

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        return batch
    

    @staticmethod
    def move_batch_to_device(batch, device: torch.device, is_train: bool):
        tensors = ['text', 'src_pos']
        if is_train:
            tensors += ['duration', 'pitch_target', 'energy_target', 'mel_pos', 'mel_target']
        for tensor_for_gpu in tensors:
            try:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
            except Exception as e: # here may be NotFoundError
                pass
        return batch

    def _evaluation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            data_list = get_data(text_cleaners=['english_cleaners'])
            for speed in [0.8, 1., 1.2]:
                for i, phn in tqdm(enumerate(data_list)):
                    mel, mel_cuda = synthesis(self.model, phn, speed)

                    self._log_spectrogram(mel_cuda.squeeze(0).cpu()) # remove batch dim
                    
                    os.makedirs("results", exist_ok=True)
                    
                    audio.tools.inv_mel_spec(
                        mel, f"results/s={speed}_{i}_{epoch}.wav"
                    )
                    
                    WaveGlow = get_WaveGlow()
                    waveglow.inference.inference(
                        mel_cuda, WaveGlow,
                        f"results/s={speed}_{i}_{epoch}_waveglow.wav"
                    )
            for filename in os.listdir(str(ROOT_PATH / 'results')):
                audio_, sr = torchaudio.load(str(ROOT_PATH / 'results' / filename))
                self._log_audio(audio_, sr, filename)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram):
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio, sample_rate, name):
        self.writer.add_audio("audio_" + name, audio, sample_rate)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
