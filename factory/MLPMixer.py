import torch.nn as nn
from einops.layers.torch import Rearrange
from functools import partial


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


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
        nn.Conv1d(num_patches, out_dim, kernel_size=5, padding=2),
    )