import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F

from hw_tts.utils.helpers_for_fastspeech import *
from hw_tts.model.common.fft import FFTBlock
from hw_tts.model.common.lr import LengthRegulator
from hw_tts.model.common.pos_encoding import DefaultPositionalEncoding

# Encoder
class Encoder(nn.Module):
    def __init__(self,
                 max_seq_len: int,
                 encoder_n_layer: int,
                 vocab_size: int,
                 encoder_dim: int,
                 pad_idx: int,
                 encoder_conv1d_filter_size: int,
                 encoder_n_head: int,
                 fft_conv1d_kernel: list,
                 fft_conv1d_padding: list,
                 dropout: float,
                 use_flash_attention: bool = False,
                 use_default_pos_encoding: bool = False):
        super().__init__()
        
        len_max_seq=max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer

        self.pad_idx = pad_idx


        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=pad_idx
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=pad_idx
        )

        if use_default_pos_encoding:
            self.position_enc = DefaultPositionalEncoding(embed_dim=encoder_dim, max_len=max_seq_len)

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_n_head,
            # encoder_dim // encoder_n_head,
            # encoder_dim // encoder_n_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout,
            use_flash_attention = use_flash_attention
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad_idx=self.pad_idx)
        non_pad_mask = get_non_pad_mask(src_seq, pad_idx=self.pad_idx)
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask





# Decoder
class Decoder(nn.Module):
    def __init__(self, 
                 max_seq_len: int,
                 decoder_n_layer: int,
                 encoder_dim: int,
                 pad_idx: int,
                 decoder_conv1d_filter_size: int,
                 decoder_n_head: int,
                 fft_conv1d_kernel: list,
                 fft_conv1d_padding: list,
                 dropout: float,
                 use_flash_attention: bool = False,
                 use_default_pos_encoding: bool = False
                 ):

        super().__init__()

        len_max_seq=max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer


        self.pad_idx = pad_idx


        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=pad_idx,
        )
        if use_default_pos_encoding:
            self.position_enc = DefaultPositionalEncoding(embed_dim=encoder_dim, max_len=max_seq_len)
            


        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            decoder_conv1d_filter_size,
            decoder_n_head,
            # encoder_dim // decoder_n_head,
            # encoder_dim // decoder_n_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout,
            use_flash_attention = use_flash_attention
        ) for _ in range(n_layers)])

        self.use_flash_attention = use_flash_attention

    def forward(self, enc_seq, enc_pos, return_attns=False):
        # print("in decoder forward")
        # print(enc_seq.shape)
        if len(enc_pos.shape) == 1:
            enc_pos = enc_pos.unsqueeze(0)
        dec_slf_attn_list = []
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad_idx=self.pad_idx)
        non_pad_mask = get_non_pad_mask(enc_pos, self.pad_idx)
        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output



class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, 
                 max_seq_len: int,
                 encoder_n_layer: int,
                 decoder_n_layer: int,
                 vocab_size: int,
                 encoder_dim: int,
                 decoder_dim: int,
                 pad_idx: int,
                 encoder_conv1d_filter_size: int,
                 encoder_n_head: int,
                 decoder_conv1d_filter_size: int,
                 decoder_n_head: int,
                 num_mels: int,
                 dropout: float,
                 fft_conv1d_kernel: list = [9, 1],
                 fft_conv1d_padding: list = [4, 0],
                 duration_predictor_filter_size = 256,
                 duration_predictor_kernel_size = 3,
                 use_flash_attention: bool = False,
                 use_default_pos_encoding: bool = False):
        super().__init__()

        self.pad_idx = pad_idx
        self.encoder = Encoder(max_seq_len,
                            encoder_n_layer,
                            vocab_size,
                            encoder_dim,
                            self.pad_idx,
                            encoder_conv1d_filter_size,
                            encoder_n_head,
                            fft_conv1d_kernel,
                            fft_conv1d_padding,
                            dropout,
                            use_flash_attention,
                            use_default_pos_encoding)
        self.length_regulator = LengthRegulator(
                            encoder_dim,
                            duration_predictor_filter_size,
                            duration_predictor_kernel_size,
                            dropout)
        self.decoder = Decoder(max_seq_len,
                            decoder_n_layer,
                            encoder_dim,
                            self.pad_idx,
                            decoder_conv1d_filter_size,
                            decoder_n_head,
                            fft_conv1d_kernel,
                            fft_conv1d_padding,
                            dropout,
                            use_flash_attention,
                            use_default_pos_encoding)

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        # src_seq: (batch_size, max_seq_len)
        # src_pos: (batch_size, max_seq_len)
        # mel_pos: (batch_size, max_mel_len)
        # mel_max_length: int
        # length_target: (batch_size, )
        # alpha: float

        # return: mel_output: (batch_size, max_mel_len, num_mels)
        # return: duration_predictor_output: (batch_size, max_seq_len)
        # return: mel_pos: (batch_size, max_mel_len)

        # -- Encode
        x, non_pad_mask = self.encoder(src_seq, src_pos, return_attns=False)

        # -- Length regulator, Decode while training
        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, length_target, mel_max_length)
            output = self.decoder(output, mel_pos)

            # mask tensor 
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output


        # -- Length regulator, Decode in inference
        output, mel_pos = self.length_regulator(x, alpha)
        output = self.decoder(output, mel_pos)
        output = self.mel_linear(output)
        return output
        
