import torch 
from torch import nn
from torch.nn import Sequential

# Duration Predictor
class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)

class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, 
                 encoder_dim,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout):
        super().__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out






# Fast Speech 2

class BasePredictor(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 filter_size: int,
                 kernel_size: int,
                 dropout: float):
        self.pipeline = Sequential(
            Transpose(-1, -2),
            nn.Conv1d(in_channels=embed_dim, out_channels=filter_size, kernel_size=kernel_size, padding=1),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
            Transpose(-1, -2),
            nn.Conv1d(in_channels=filter_size, out_channels=filter_size, kernel_size=kernel_size, padding=1),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
            nn.Linear(filter_size, 1)
        )


    def forward(self, x):
        return self.pipeline(x)
    

class PitchPredictor(nn.Module):
    '''
    To take the pitch contour as input in both training and inference,
    we quantize pitch F0 (ground-truth/predicted value for train/inference respectively) of each frame
    to 256 possible values in log-scale and further convert it into pitch embedding vector p and add it to
    the expanded hidden sequence.
    '''
    def __init__(self,
                 vocab_size: int,
                 max_pitch: float, # we need this param for scaling 
                 embed_dim: int,
                 filter_size: int,
                 kernel_size: int,
                 dropout: float):
        self.predictor = BasePredictor(embed_dim,
                                       filter_size,
                                       kernel_size,
                                       dropout)
        self.max_pitch = torch.tensor(max_pitch)
        self.vocab_size = vocab_size
        self.embeds = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x, target=None, beta=1.0):
        pitch_preds = self.predictor(x)

        linspace = torch.linspace(0, torch.log1p(self.max_pitch), self.vocab_size)
        if target is not None:
            log_target = torch.log1p(target)
            buckets = torch.bucketize(log_target, linspace)
        else:
            pitch_preds = torch.exp(pitch_preds) * beta
            log_preds = torch.log1p(pitch_preds)
            buckets = torch.bucketize(log_preds, linspace)
        return self.embeds(buckets), pitch_preds
    

class EnergyPredictor(nn.Module):
    '''
    To take the pitch contour as input in both training and inference,
    we quantize pitch F0 (ground-truth/predicted value for train/inference respectively) of each frame
    to 256 possible values in log-scale and further convert it into pitch embedding vector p and add it to
    the expanded hidden sequence.
    '''
    def __init__(self,
                 vocab_size: int,
                 max_energy: float, # we need this param for scaling 
                 embed_dim: int,
                 filter_size: int,
                 kernel_size: int,
                 dropout: float):
        self.predictor = BasePredictor(embed_dim,
                                       filter_size,
                                       kernel_size,
                                       dropout)
        self.max_energy = torch.tensor(max_energy)
        self.vocab_size = vocab_size
        self.embeds = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x, target=None, gamma=1.0):
        energy_preds = self.predictor(x)

        linspace = torch.linspace(0, torch.log1p(self.max_energy), self.vocab_size)
        if target is not None:
            log_target = torch.log1p(target)
            buckets = torch.bucketize(log_target, linspace)
        else:
            energy_preds = torch.exp(energy_preds) * gamma
            log_preds = torch.log1p(energy_preds)
            buckets = torch.bucketize(log_preds, linspace)
        return self.embeds(buckets), energy_preds

