import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **batch):
        mel_output, mel_target = batch["mel_output"], batch["mel_target"]
        print("mel shapes", mel_output.shape, mel_target.shape)
        duration_predictor_output, length_target = batch["duration_predictor_output"], batch["duration"]
        print("duration shapes", duration_predictor_output.shape, length_target.shape)
        mel_loss = self.mse(mel_output, mel_target)


        # we compare the log of the length_target with the duration_predictor_output
        # because we predict log of real duration
        duration_predictor_loss = self.mse(duration_predictor_output,
                                                torch.log((length_target + 1).float())) 


        return mel_loss, duration_predictor_loss, None, None