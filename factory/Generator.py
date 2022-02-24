import torch.nn as nn
from .MLPMixer import MLPMixer
from .Norm import GroupNorm, ConvNorm, AdaIN


class MetaBlock(nn.Module):
    def __init__(
        self,
        dim,
        source_emb=88,
        crop_len=176,
        patch_size=8,
        mlp_depth=1,
        conv_activate=nn.ReLU(),
        w_init_gain="relu",
        norm_layer=GroupNorm,
    ):

        super(MetaBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = ConvNorm(
            source_emb, source_emb, kernel_size=5, padding=2, w_init_gain=w_init_gain
        )
        self.norm2 = norm_layer(crop_len)
        self.conv_1 = ConvNorm(
            source_emb, crop_len, kernel_size=5, padding=2, w_init_gain=w_init_gain
        )
        self.mlp = MLPMixer(
            image_size=crop_len,
            channels=1,
            patch_size=patch_size,
            dim=crop_len,
            depth=mlp_depth,
            out_dim=source_emb,
        )
        self.conv_2 = ConvNorm(
            source_emb,
            source_emb,
            kernel_size=5,
            padding=2,
            w_init_gain=w_init_gain,
        )
        self.activate = conv_activate
        self.adain = AdaIN()

    def forward(self, x, embedding):

        # Part1 --- Token Mixer
        x = x + self.token_mixer(self.norm1(x))
        x = self.adain(x, embedding)
        x = self.activate(x)
        # Part2 --- Middle Conv
        x_after_conv_1 = self.conv_1(x)
        x_after_conv_1 = self.adain(x_after_conv_1, embedding)
        x_after_conv_1 = self.activate(x_after_conv_1)
        # Part3 --- MLP
        x_mlp = self.mlp(self.norm2(x_after_conv_1).unsqueeze(1))
        # Part4 --- Last Conv
        x_after_conv_2 = self.conv_2(x_mlp)
        x_after_conv_2 = self.adain(x_after_conv_2, embedding)
        x_after_conv_2 = self.activate(x_after_conv_2)

        return x + x_after_conv_2


class Generator(nn.Module):
    def __init__(
        self,
        crop_len=176,
        dim_neck=44,
        num_layers=3,
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
            out_dim=2 * dim_neck,
        )

    def forward(self, x, embedding):
        for _ in range(self.num_layer):
            x = self.metablock(x, embedding)
        x = self.conv(x)
        x = self.adain(x, embedding)
        x = self.activate(x)
        x = self.mlp(x.unsqueeze(1))
        return x