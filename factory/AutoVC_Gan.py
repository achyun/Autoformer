from calendar import c
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Adjust import Adjust
from .Norm import ConvNorm, LinearNorm


class Encoder(nn.Module):
    """Encoder module:"""

    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    80 + dim_emb if i == 0 else 512,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(512),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

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

    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)

        convolutions = []
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    dim_pre,
                    dim_pre,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(dim_pre),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)

        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):

        # self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        decoder_output = self.linear_projection(outputs)

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

        for _ in range(1, 5 - 1):
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


class AutoVC_Gan(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(AutoVC_Gan, self).__init__()
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        # 這個是拿來校正 feature distangle 的 speaker embedding 用的
        self.adjust = Adjust(dim_emb)

    def concatenate(self, x, emb):
        x = x.squeeze(1).transpose(2, 1)
        emb = emb.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return torch.cat((x, emb), dim=1)

    def get_code(self, x, c_org, use_adjust):
        if use_adjust:
            # adjust speaker embedding for encoder, during traning c_org is equal to c_trg
            c_org = self.adjust(self.concatenate(x, c_org))
            # Notice! Here is original x with adjust speaker embedding
        return self.encoder(x, c_org), c_org

    def get_content_with_code(self, codes, size):
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(size / len(codes)), -1))

        return torch.cat(tmp, dim=1)

    def get_decode(self, contents_code):
        mel_outputs = self.decoder(contents_code)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        return mel_outputs, mel_outputs_postnet

    def forward(self, x, c_org, c_trg, use_adjust=False):

        codes, c_org = self.get_code(x, c_org, use_adjust)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        else:
            x_t = self.concatenate(x, c_trg)
            c_trg = self.adjust(x_t)
            contents_code = self.get_content_with_code(codes, x.size(1))
            cat_code_emb = torch.cat(
                (contents_code, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
            )
            mel_outputs, mel_outputs_postnet = self.get_decode(cat_code_emb)
            return c_org, mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
