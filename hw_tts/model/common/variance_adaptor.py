from hw_tts.model.common.predictors import PitchPredictor, EnergyPredictor
from hw_tts.model.common.lr import LengthRegulator

import torch
from torch import nn

class VarianceAdaptor(nn.Module):
    def __init__(self, 
                 energy_vocab_size: int, 
                 pitch_vocab_size: int, 
                 max_energy: float,
                 max_pitch: float,
                 embed_dim: int,
                 filter_size: int,
                 kernel_size: int,
                 dropout: float):
        self.length_regulator = LengthRegulator(embed_dim,
                                                filter_size,
                                                kernel_size,
                                                dropout)
        self.energy_predictor = EnergyPredictor(energy_vocab_size,
                                                max_energy,
                                                embed_dim,
                                                filter_size,
                                                kernel_size,
                                                dropout)
        self.pitch_predictor = PitchPredictor(pitch_vocab_size,
                                                max_pitch,
                                                embed_dim,
                                                filter_size,
                                                kernel_size,
                                                dropout)
        


    def forward(self,
                x,
                length_target=None,
                mel_max_length=None,
                alpha: float = 1.0,
                pitch_target=None,
                beta: float = 1.0,
                energy_target=None,
                gamma: float = 1.0):
        if self.training:
            pitch_embeds, pitch_preds,  = self.pitch_predictor(x, pitch_target, beta)
            energy_embeds, energy_preds = self.energy_predictor(x, energy_target, gamma)
            x, duration_preds = self.length_regulator(x, alpha, length_target, mel_max_length)
            return x + pitch_embeds + energy_embeds, duration_preds, pitch_preds, energy_preds
        
    # make forward pass for all predictors, add the embeddings together
    # TODO: return final embeds, mel_pos, all predictions
        
