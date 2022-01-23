# -*- coding: utf-8 -*-
from decimal import DivisionImpossible
from functools import partial

import torch
from torch._C import wait
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
    *,
    image_size,
    channels,
    patch_size,
    dim,
    depth,
    out_dim,
    expansion_factor=4,
    dropout=0.0
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
        nn.Conv1d(num_patches, out_dim, kernel_size=3, padding=1),
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
    def __init__(
        self,
        dim=176,
        patch_size=16,
        mlp_depth=1,
        kernel_size=5,
        channels=1,
        norm_layer=GroupNorm,
        conv_activate=nn.ReLU(),
        w_init_gain="relu",
    ):
        super().__init__()
        self.emb = PatchEmbed(in_chans=dim, embed_dim=dim)
        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.token_mixer =  nn.Sequential(
                ConvNorm(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=2,
                    w_init_gain=w_init_gain,
                ),
                nn.BatchNorm1d(dim),
                conv_activate,
            )
        self.mlp = MLPMixer(
            image_size=dim,
            channels=channels,
            patch_size=patch_size,
            dim=dim,
            depth=mlp_depth,
            out_dim=dim,
        )

    def forward(self, x):
        x = self.emb(x)
        x = x + self.token_mixer(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x).unsqueeze(1))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dim_neck,
        dim_emb,
        freq,
        crop_len=176,
        num_layers=3,
    ):
        super().__init__()
        self.freq = freq
        self.dim_neck = dim_neck
        self.conv1 = ConvNorm(
            80 + dim_emb, crop_len, kernel_size=5, padding=2, w_init_gain="relu"
        )
        self.bn1 = nn.BatchNorm1d(crop_len)
        network = []
        for _ in range(num_layers):
            network.append(MetaBlock(dim=crop_len))
        self.metablock = nn.Sequential(*network)
        self.mlp = MLPMixer(
            image_size=176,
            channels=1,
            patch_size=4,
            dim=176,
            out_dim=2 * dim_neck,
            depth=3,
        )

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        # (b,lencrop,lencrop)
        x = self.metablock(x)
        # (b,lencrop,lencrop)
        x = self.mlp(x.unsqueeze(1))
        # (b,2*dim_neck,lencrop)
        outputs = x.transpose(1, 2)
        # OUT (batch, lencrop, 2*dim_neck)
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
    def __init__(self, dim_neck, dim_emb, crop_len=176, num_layers=2):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb,512,1, batch_first=True)
        self.conv1 = ConvNorm(
            crop_len, 512, kernel_size=5, padding=2, w_init_gain="relu"
        )
        self.bn1 = nn.BatchNorm1d(512)
        network = []
        for _ in range(num_layers):
            network.append(MetaBlock(dim=512))
        self.metablock = nn.Sequential(*network)
        self.mlp = MLPMixer(
            image_size=512, channels=1, patch_size=128, dim=512, out_dim=1024, depth=2
        )
        self.conv2 = ConvNorm(512, 176, kernel_size=5, padding=2, w_init_gain="relu")
        self.bn2 = nn.BatchNorm1d(176)
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        # (b,512,512)
        x = self.metablock(x)
        # (b,512,512)
        x = self.mlp(x.unsqueeze(1))
        # (b,512,1024)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # (b,176,1024)
        x = self.linear_projection(x)
        # (b,176,80)
        return x


class Postnet(nn.Module):
    def __init__(self, num_layers=1):
        super(Postnet, self).__init__()
        self.conv1 = ConvNorm(80, 176, kernel_size=5, padding=2, w_init_gain="tanh")
        self.bn1 = nn.BatchNorm1d(176)
        network = []
        for _ in range(num_layers):
            network.append(
                MetaBlock(dim=176, conv_activate=nn.Tanh(), w_init_gain="tanh")
            )
        self.metablock = nn.Sequential(*network)
        self.conv2 = ConvNorm(176, 80, kernel_size=5, padding=2, w_init_gain="linear")
        self.bn2 = nn.BatchNorm1d(80)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(self.bn1(x))
        # (b,176,176)
        x = self.metablock(x)
        # (b,176,176)
        x = self.conv2(x)
        x = self.bn2(x)
        # (b,80,176)
        return x


class AutoVC_MetaFormer_CONV_V2(nn.Module):
    """AutoVC_MLP_MIXER network."""

    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(AutoVC_MetaFormer_CONV_V2, self).__init__()
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb)
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
