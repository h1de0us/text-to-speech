import torch
from torch import nn


# Vanilla Positional Encoding
class DefaultPositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print(pe.shape, position.shape
        pe[:, 0::2] = torch.sin(position / torch.pow(10000, torch.arange(0, embed_dim, 2).float() / embed_dim))
        pe[:, 1::2] = torch.cos(position / torch.pow(10000, torch.arange(0, embed_dim, 2).float() / embed_dim))
        pe = pe.unsqueeze(0)
        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
