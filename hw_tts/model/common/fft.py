import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, 
                 d_in, 
                 d_hid, 
                 fft_conv1d_kernel: list,
                 fft_conv1d_padding: list,
                 dropout=0.1):
        super().__init__()

        # Use Conv1D instead of Linear
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                #  d_k,
                #  d_v,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1,
                 use_flash_attention=False):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout, 
            batch_first=True)
            # kdim=d_k, 
            # vdim=d_v)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, 
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout)
        self.use_flash_attention = use_flash_attention
        
    def _forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # print('mask.shape', slf_attn_mask.shape) # (bs, seq_len, seq_len)
        # slf_attn_mask is actually a mask for keys
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, key_padding_mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        if self.use_flash_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
            ):
                return self._forward(enc_input, non_pad_mask, slf_attn_mask)
        
        return self._forward(enc_input, non_pad_mask, slf_attn_mask)


