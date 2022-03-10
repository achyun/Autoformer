import torch
import torch.nn as nn
import random
from functools import partial
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
        num_classes,
        dim_emb=256,
        emb=768,
        crop_len=128,
        patch_size=32,
        mlp_depth=1,
        kernel_size=3,
        channels=1,
        sample_count = 4,
        norm_layer=nn.BatchNorm1d,
    ):
 
        super().__init__()
        # 這個 token_mixer 可以換成其他的，就是 "MetaFormer" 的概念
        self.token_mixer = nn.LSTM(
            input_size=80,
            hidden_size=emb,
            num_layers=1,
            batch_first=True,
        )
        self.conv_1 = ConvNorm(crop_len, emb, kernel_size=kernel_size)
        self.mlp = MLPMixer(
            image_size=emb,
            channels=channels,
            patch_size=patch_size,
            dim=emb,
            depth=mlp_depth,
        )
        self.dv = nn.Linear(emb * 2, dim_emb)
        self.output = nn.Linear(dim_emb, num_classes)
        self.crop_len = crop_len
        self.sample_count = sample_count

        
    def forward(self, x):
        x, _ = self.token_mixer(x)
        embedding_1 = x[:, -1, :]
        embedding_2 = []
        for i in range(self.sample_count):
            left = random.randint(0, x.shape[1] - self.crop_len)
            x_after_conv_1 = self.conv_1(x[:,left : left + self.crop_len,:])
            x_mlp = self.mlp(x_after_conv_1.unsqueeze(1))
            embedding_2.append(x_mlp[:, -1, :])
        
        embedding_2 = torch.stack(embedding_2).sum(dim=0).div(self.sample_count)
        embedding = self.dv(torch.cat((embedding_1, embedding_2), dim=1))
        norm = embedding.norm(p=2, dim=-1, keepdim=True)
        d_vec = embedding.div(norm)
        predictions = self.output(embedding)
        return predictions, d_vec


class MetaDV(nn.Module):
    def __init__(self, num_classes=256):
        super(MetaDV, self).__init__()
        self.metablock = MetaBlock(num_classes)

    def forward(self, x):
        return self.metablock(x)

