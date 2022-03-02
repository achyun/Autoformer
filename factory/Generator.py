import torch.nn as nn
from .MLPMixer import MLPMixer
from .Norm import ConvNorm, AdaIN


class MetaBlock(nn.Module):
    def __init__(
        self,
        dim=88,
        crop_len=176,
        patch_size=8,
        mlp_depth=1,
        conv_activate=nn.ReLU(),
        w_init_gain="relu",
    ):

        super(MetaBlock, self).__init__()

        self.token_mixer = ConvNorm(
            dim, dim, kernel_size=5, padding=2, w_init_gain=w_init_gain
        )

        self.conv_1 = ConvNorm(
            dim, crop_len, kernel_size=5, padding=2, w_init_gain=w_init_gain
        )
        self.mlp = MLPMixer(
            image_size=crop_len,
            channels=1,
            patch_size=patch_size,
            dim=crop_len,
            depth=mlp_depth,
            out_dim=dim,
        )
        self.conv_2 = ConvNorm(
            dim, dim, kernel_size=5, padding=2, w_init_gain=w_init_gain,
        )
        self.activate = conv_activate
        self.adain = AdaIN()

    def forward(self, x, embedding):

        # Part1 --- Token Mixer
        x = x.transpose(1, 2)
        x = x + self.token_mixer(x)
        x = self.adain(x, embedding)
        x = self.activate(x)
        # Part2 --- Middle Conv
        x_after_conv_1 = self.conv_1(x)
        x_after_conv_1 = self.adain(x_after_conv_1, embedding)
        x_after_conv_1 = self.activate(x_after_conv_1)
        # Part3 --- MLP
        x_mlp = self.mlp(x_after_conv_1.unsqueeze(1))
        # Part4 --- Last Conv
        x_after_conv_2 = self.conv_2(x_mlp)
        x_after_conv_2 = self.adain(x_after_conv_2, embedding)
        x_after_conv_2 = self.activate(x_after_conv_2)

        return (x + x_after_conv_2).transpose(1, 2)


class Generator(nn.Module):
    def __init__(
        self, crop_len=176, dim_neck=44, out_mel=80, num_layers=3,
    ):
        super(Generator, self).__init__()

        self.num_layer = num_layers
        self.metablock = MetaBlock(2 * dim_neck)
        self.conv = ConvNorm(
            2 * dim_neck, crop_len, kernel_size=5, padding=2, w_init_gain="relu"
        )
        self.adain = AdaIN()
        self.activate = nn.ReLU()

        self.mlp = MLPMixer(
            image_size=crop_len,
            channels=1,
            patch_size=16,
            dim=crop_len,
            depth=1,
            out_dim=out_mel,
        )

    def forward(self, x, embedding):
        # x --- (b,176,2*dim_neck)
        for _ in range(self.num_layer):
            x = self.metablock(x, embedding)
        # (b,176,88)
        x = self.conv(x.transpose(1, 2))
        # (b,176,176)
        x = self.adain(x, embedding)
        x = self.activate(x)
        x = self.mlp(x.unsqueeze(1)).transpose(1, 2)
        # (b,176,80)

        return x
