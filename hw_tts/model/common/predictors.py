import torch 
from torch import nn


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
    def __init__(self, *args, **kwargs):
        self.conv1 = nn.Conv1d(...)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(...)
        self.dropout1 = nn.Dropout(...)
        self.conv2 = nn.Conv1d(...)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LayerNorm(...)
        self.dropout2 = nn.Dropout(...)
        self.linear = nn.Linear(...)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.linear(x)
        return x