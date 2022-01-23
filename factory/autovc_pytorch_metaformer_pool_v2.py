# -*- coding: utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )


def MLPMixer(
    *, image_size, channels, patch_size, dim, depth, expansion_factor=4, dropout=0.0
):
    assert (image_size % patch_size) == 0, "image must be divisible by patch size"
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        ),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[
            nn.Sequential(
                PreNormResidual(
                    dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)
                ),
                PreNormResidual(
                    dim, FeedForward(dim, expansion_factor, dropout, chan_last)
                ),
            )
            for _ in range(depth)
        ],
        nn.Conv1d(num_patches, 88, kernel_size=3, padding=1),
    )


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        patch_size=5,
        stride=1,
        padding=2,
        in_chans=336,
        embed_dim=512,
    ):
        super().__init__()
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class MetaBlock(nn.Module):
    """"""

    def __init__(
        self,
        dim,
        source_emb = 512,
        crop_len = 176,
        out_dim_neck = 88,
        patch_size = 8,
        mlp_depth = 1,
        pool_size=3,
        norm_layer=GroupNorm,
    ):

        super().__init__()
        self.norm1 = norm_layer(dim)
        # 這個 token_mixer 可以換成其他的，就是 "MetaFormer" 的概念
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(crop_len)
        self.conv_1 = ConvNorm(source_emb, crop_len, kernel_size=3)
        self.mlp = MLPMixer(image_size=crop_len, channels=1, patch_size=patch_size, dim=crop_len, depth=mlp_depth)
        self.conv_2 = ConvNorm(out_dim_neck,source_emb, kernel_size=3)

    def forward(self, x):
        # (batch, 512, 176)
        x = x + self.token_mixer(self.norm1(x))
        # (batch, 512, 176)
        x_after_conv_1 = self.conv_1(x)
        # (batch, 176, 176)
        x_mlp = self.mlp(self.norm2(x_after_conv_1).unsqueeze(1))
        # (batch, 88, 176)
        x_after_conv_2 = self.conv_2(x_mlp)
        # (batch, 512, 176)
        x = x + x_after_conv_2
        # (batch, 512, 176)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dim_neck,
        freq,
        dim,
        num_layers=6,
    ):
        super().__init__()
        self.freq = freq
        self.dim_neck = dim_neck
        self.embding = PatchEmbed()
        network = []
        for i in range(num_layers):
            network.append(MetaBlock(dim))
        self.metablock = nn.Sequential(*network)
        self.output_conv = ConvNorm(512, 176, kernel_size=3)
        self.mlp = MLPMixer(image_size=176, channels=1, patch_size=16, dim=176, depth=1)

    def forward(self, x, c_org):

        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        # (batch,336,176)
        x = self.embding(x)
        # (batch,512,176)
        x = self.metablock(x)
        # (batch, 512, 176)
        x = self.output_conv(x)
        # (batch, 176, 176)
        x = x.unsqueeze(1)
        x = self.mlp(x)
        # OUT (batch, lencrop, 2*self.dim_neck)
        outputs = x.transpose(1, 2)
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck :]
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(
                torch.cat(
                    (out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]),
                    dim=-1,
                )
            )
        return codes


class Decoder(nn.Module):
    """Decoder module:"""

    def __init__(self,dim,num_layers=3):
        super(Decoder, self).__init__()
        self.embding = PatchEmbed(in_chans=176)
        network = []
        for i in range(num_layers):
            network.append(MetaBlock(dim,crop_len=344,patch_size=8))
        self.metablock = nn.Sequential(*network)
        self.output_conv_1 = ConvNorm(512, 344, kernel_size=3)
        self.mlp = MLPMixer(image_size=344, channels=1, patch_size=8, dim=344, depth=1)
        self.output_conv_2 = ConvNorm(344,176, kernel_size=3)
        self.linear_projection = LinearNorm(88, 80)

    def forward(self, x):
        # (batch,176,344)
        x = self.embding(x)
        # (batch,512,344)
        x = self.metablock(x)
        # (batch, 512, 344)
        x = self.output_conv_1(x)
        # (batch, 344, 344)
        x = x.unsqueeze(1)
        x = self.mlp(x)
        # (batch, 2*self.dim_neck, 344)
        x = x.transpose(2,1)
        # (batch,  344, 2*self.dim_neck)
        x = self.output_conv_2(x)
        # (batch, 176, 2*self.dim_neck)
        decoder_output = self.linear_projection(x)
        # (batch,176, 80)
        return decoder_output


class Postnet(nn.Module):
    """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    80,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(512),
            )
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        512,
                        512,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(512),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    512,
                    80,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(80),
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


class AutoVC_MetaFormer_Pool_V2(nn.Module):
    """AutoVC_MLP_MIXER network."""

    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(AutoVC_MetaFormer_Pool_V2, self).__init__()
        self.encoder = Encoder(dim_neck, freq, dim_pre)
        self.decoder = Decoder(dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):

        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat(
            (code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
        )
        mel_outputs = self.decoder(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

