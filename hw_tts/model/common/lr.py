
import torch 
from torch import nn
from torch.nn import functional as F

from .predictors import DurationPredictor
from hw_tts.utils.helpers_for_fastspeech import create_alignment

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self,
                 encoder_dim,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout):
        super().__init__()
        self.duration_predictor = DurationPredictor(encoder_dim,
                                                    duration_predictor_filter_size,
                                                    duration_predictor_kernel_size,
                                                    dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            # if we have target, we also have mel_max_length
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        # return new positional embeddings and dublicated output
        duration_predictor_output = (duration_predictor_output * alpha + 0.5).int()
        output = self.LR(x, duration_predictor_output)
        mel_pos = torch.arange(1, output.size(1) + 1, dtype=torch.long, device=x.device)
        return output, mel_pos
