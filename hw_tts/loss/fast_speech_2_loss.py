import torch
from torch import nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, 
                **batch):
        mel_output = batch["mel_output"]
        duration_predictor_output = batch["duration_predictor_output"]
        pitch_predictor_output = batch["pitch_predictor_output"]    
        energy_predictor_output = batch["energy_predictor_output"]
        mel_target = batch["mel_target"]
        length_target = batch["length_target"]
        pitch_target = batch["pitch_target"]
        energy_target = batch["energy_target"]

        mel_loss = self.mse(mel_output, mel_target)

        duration_predictor_loss = self.mse(duration_predictor_output,
                                                torch.log((length_target + 1).float())) 

        energy_predictor_loss = self.mse(energy_predictor_output,
                                              torch.log(energy_target + 1))

        pitch_predictor_loss = self.mse(pitch_predictor_output,
                                              torch.log(pitch_target + 1))

        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss