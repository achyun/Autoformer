import torch
import torch.nn as nn

from .MLPMixer import MLPMixer
from .Norm import ConvNorm, GroupNorm, LinearNorm, PatchEmbed


class MetaBlock(nn.Module):
    def __init__(
        self,
        dim,
        source_emb=512,
        crop_len=176,
        out_dim_neck=88,
        patch_size=8,
        mlp_depth=1,
        conv_activate=nn.ReLU(),
        w_init_gain="relu",
        norm_layer=GroupNorm,
    ):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = nn.Sequential(
            ConvNorm(
                source_emb,
                source_emb,
                kernel_size=5,
                padding=2,
                w_init_gain=w_init_gain,
            ),
            nn.BatchNorm1d(source_emb),
            conv_activate,
        )
        self.norm2 = norm_layer(crop_len)
        self.conv_1 = nn.Sequential(
            ConvNorm(
                source_emb, crop_len, kernel_size=5, padding=2, w_init_gain=w_init_gain,
            ),
            nn.BatchNorm1d(crop_len),
            conv_activate,
        )

        self.mlp = MLPMixer(
            image_size=crop_len,
            channels=1,
            patch_size=patch_size,
            dim=crop_len,
            depth=mlp_depth,
            out_dim=out_dim_neck,
        )
        self.conv_2 = nn.Sequential(
            ConvNorm(
                out_dim_neck,
                source_emb,
                kernel_size=5,
                padding=2,
                w_init_gain=w_init_gain,
            ),
            nn.BatchNorm1d(source_emb),
            conv_activate,
        )

    def forward(self, x):
        # (batch, 512, 176)
        x = x + self.token_mixer(self.norm1(x))
        # (batch, 512, 176)
        x_after_conv_1 = self.conv_1(x)
        # (batch, 176, 176)
        x_mlp = self.mlp(self.norm2(x_after_conv_1).unsqueeze(1))
        # (batch, out_dim_neck, 176)
        x_after_conv_2 = self.conv_2(x_mlp)
        # (batch, 512, 176)
        x = x + x_after_conv_2
        # (batch, 512, 176)
        return x


class Encoder(nn.Module):
    def __init__(
        self, dim_neck, freq, dim, num_layers=3,
    ):
        super().__init__()
        self.freq = freq
        self.dim_neck = dim_neck
        self.embding = PatchEmbed()
        network = []
        for _ in range(num_layers):
            network.append(MetaBlock(dim))
        self.metablock = nn.Sequential(*network)
        self.output_conv = nn.Sequential(
            ConvNorm(512, 176, kernel_size=5, padding=2, w_init_gain="relu"),
            nn.BatchNorm1d(176),
            nn.ReLU(),
        )

        self.mlp = MLPMixer(
            image_size=176,
            channels=1,
            patch_size=16,
            dim=176,
            depth=1,
            out_dim=2 * dim_neck,
        )

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
    def __init__(self, dim, num_layers=1):
        super(Decoder, self).__init__()
        self.embding = PatchEmbed(in_chans=176)
        network = []
        for i in range(num_layers):
            network.append(MetaBlock(dim, crop_len=344, patch_size=8, out_dim_neck=88))
        self.metablock = nn.Sequential(*network)

        self.output_conv_1 = nn.Sequential(
            ConvNorm(512, 344, kernel_size=5, padding=2, w_init_gain="relu"),
            nn.BatchNorm1d(344),
            nn.ReLU(),
        )

        self.mlp = MLPMixer(
            image_size=344, channels=1, patch_size=8, dim=344, depth=1, out_dim=88
        )

        self.output_conv_2 = nn.Sequential(
            ConvNorm(344, 176, kernel_size=5, padding=2, w_init_gain="relu"),
            nn.BatchNorm1d(176),
            nn.ReLU(),
        )

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
        # (batch, 176, 344)
        x = x.transpose(2, 1)
        # (batch,  344, 176)
        x = self.output_conv_2(x)
        # (batch, 176, 176)
        decoder_output = self.linear_projection(x)
        # (batch,176, 80)
        return decoder_output


class Postnet(nn.Module):
    """
    Postnet
    - Same as AutoVC
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


class MetaConv(nn.Module):
    def __init__(self, dim_neck, dim, dim_pre, freq):
        super(MetaConv, self).__init__()
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
