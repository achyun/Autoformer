import torch.nn as nn
from .MLPMixer import MLPMixer
from .Norm import ConvNorm, AdaIN


class MetaBlock(nn.Module):
    def __init__(
        self, mel_dim=80, crop_len=176, patch_size=8, mlp_depth=1, w_init_gain="relu",
    ):

        super(MetaBlock, self).__init__()

        self.token_mixer = ConvNorm(
            mel_dim, mel_dim, kernel_size=5, padding=2, w_init_gain=w_init_gain
        )

        self.conv = ConvNorm(
            mel_dim, crop_len, kernel_size=5, padding=2, w_init_gain=w_init_gain
        )
        self.adain = AdaIN()
        self.activate = nn.ReLU()
        self.mlp = MLPMixer(
            image_size=crop_len,
            channels=1,
            patch_size=patch_size,
            dim=crop_len,
            depth=mlp_depth,
            out_dim=mel_dim,
        )

    def forward(self, x, embedding):

        # Part1 --- Token Mixer
        x = x.transpose(1, 2)
        x = x + self.token_mixer(x)
        x = self.adain(x, embedding)
        x = self.activate(x)
        # Part2 --- Middle Conv
        x_after_conv = self.conv(x)
        x_after_conv = self.adain(x_after_conv, embedding)
        x_after_conv = self.activate(x_after_conv)
        # Part3 --- MLP
        x_mlp = self.mlp(x_after_conv.unsqueeze(1))
        return (x + x_mlp).transpose(1, 2)


class Patcher(nn.Module):
    def __init__(
        self, crop_len=176, out_mel=80, num_layers=3,
    ):
        super(Patcher, self).__init__()

        self.num_layer = num_layers
        self.metablock = MetaBlock()

        self.mlp = MLPMixer(
            image_size=crop_len,
            channels=1,
            patch_size=16,
            dim=crop_len,
            depth=1,
            out_dim=out_mel,
        )

    def forward(self, x, embedding):
        # x --- (b,176,80)
        for _ in range(self.num_layer):
            x = self.metablock(x, embedding)
        # x --- (b,176,80)
        return x
