import importlib
from this import d
import time
import torch
import datetime
import torch.nn.functional as F
import wandb
import os
import argparse
from torch.backends import cudnn
from util.data_loader import get_loader
from factory.Generator import Generator
from factory.Discriminator import Discriminator
from factory.Adjust import Adjust


class Solver(object):
    """
    AutoVC 加上 Adjust Speaker Embedding
    """

    def __init__(self, vcc_loader, style_loader, config):
        """Initialize configurations."""

        # Data loader and All Speaker style
        self.vcc_loader = vcc_loader
        self.style_loader = style_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.lambda_ad = config.lambda_ad
        self.lambda_cls = config.lambda_cls
        self.lambda_dis = config.lambda_dis
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.n_critic = config.n_critic
        self.model_name = config.model_name

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.log_step = config.log_step

        # Build the model.
        self.build_model()

    def build_model(self):

        self.VC = getattr(
            importlib.import_module(f"factory.{self.model_name}"), self.model_name
        )(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)

        self.vc_optimizer = torch.optim.Adam(self.VC.parameters(), 0.0001)
        self.VC.to(self.device)

        # 這個是拿來分類轉換後聲音的，算 speaker embedding 之間的 cos-similiarty
        self.C = Adjust(self.dim_emb)
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), 0.0001)
        self.C.to(self.device)

        # 拿 VC 生成的 mel 進 Generator 跟不同的 speaker embedding 生成新的聲音
        self.G = Generator()
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        self.G.to(self.device)

        # 判斷重建後的聲音真假
        self.D = Discriminator()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), 0.0001)
        self.D.to(self.device)

    def reset_grad(self):
        self.vc_optimizer.zero_grad()
        self.c_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def discriminator_loss(self, real, fake):
        real_loss = torch.nn.BCELoss()(real, torch.ones_like(real))
        fake_loss = torch.nn.BCELoss()(fake, torch.zeros_like(fake))
        return real_loss + fake_loss

    def train(self):

        data_loader = self.vcc_loader
        style_loader = self.style_loader
        keys = [
            "G/loss_id",
            "G/loss_id_psnt",
            "G/loss_cd",
            "A/loss_adjust",
            "C/loss_trans",
            "C/loss_reconst",
            "D/loss",
        ]

        print("Start training...")
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, org_style = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, org_style = next(data_iter)

            try:
                target_style = next(style_loader)
            except:
                style_iter = iter(style_loader)
                target_style = next(style_iter)

            x_real = x_real.to(self.device)
            org_style = org_style.to(self.device)
            target_style = target_style.to(self.device)

            # =================================================================================== #
            #                       2. Train the VC and Generator                                 #
            # =================================================================================== #
            self.VC = self.VC.train()
            self.C = self.C.train()
            self.G = self.G.train()
            self.D = self.D.train()
            org_style_adjust, x_identic, x_identic_psnt, code_real = self.VC(
                x_real, org_style, org_style
            )
            code_reconst = self.VC(x_identic_psnt, org_style, None)
            # Identity mapping loss
            vc_loss_id = F.mse_loss(x_real, x_identic.squeeze())
            vc_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())
            # Code semantic loss.
            g_loss_cd = F.l1_loss(code_real, code_reconst)
            # Adjust Speaker embedding loss
            a_loss_adjust = F.l1_loss(org_style_adjust, org_style)
            vc_loss = (
                vc_loss_id
                + vc_loss_id_psnt
                + self.lambda_cd * g_loss_cd
                + self.lambda_ad * a_loss_adjust
            )
            self.reset_grad()
            vc_loss.backward()
            self.vc_optimizer()

            if (i + 1) % self.n_critic == 0:
                # 用 Encoder 把 VC 自己轉自己的聲音萃取出只有 content 的內容 -> (Batch_size,Crop_len,2*dim_neck)
                source_codes = self.VC.get_code(
                    x_identic_psnt.transpose(1, 2), org_style
                )
                source_content = self.VC.get_content_with_code(source_codes)
                # Rebuild Voice with Source Content and Target style
                trans_voice = self.G(source_content, target_style)
                # 用 Encoder 把 Generator 轉換出來的聲音萃取出只有 content 的內容 -> (Batch_size,Crop_len,2*dim_neck)
                trans_codes = self.VC.get_code(trans_voice, target_style)
                trans_content = self.VC.get_content_with_code(trans_codes)
                # Recontruct Voice with Trans Content and Org style
                reconstruct_voice = self.G(trans_content, org_style)
                # 抽出 style
                trans_style = self.C(trans_voice)
                reconstruct_style = self.C(reconstruct_voice)
                # Prob
                real_prob = self.D(x_real)
                fake_prob = self.D(reconstruct_voice.squeeze())
                # Classifier loss
                c_loss_trans = F.cosine_embedding_loss(
                    trans_style, target_style, torch.tensor([1, 1])
                )
                c_loss_recontruct = F.cosine_embedding_loss(
                    reconstruct_style, org_style, torch.tensor([1, 1])
                )
                # Discriminator loss
                d_loss = self.discriminator_loss(real_prob, fake_prob)
                g_loss = (
                    self.lambda_cls * c_loss_trans
                    + self.lambda_cls * c_loss_recontruct
                    + self.lambda_dis * d_loss
                )
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                self.c_optimizer.step()
                self.d_optimizer.step()
                loss["C/loss_trans"] = c_loss_trans.item()
                loss["C/loss_reconst"] = c_loss_recontruct.item()
                loss["D/loss"] = d_loss.item()

            # Logging for VC.
            loss = {}
            loss["VC/loss_id"] = vc_loss_id.item()
            loss["VC/loss_id_psnt"] = vc_loss_id_psnt.item()
            loss["VC/loss_cd"] = g_loss_cd.item()
            loss["A/loss_adjust"] = a_loss_adjust.item()

            # =================================================================================== #
            #                               4. Print Traning Info                                 #
            # =================================================================================== #

            if (i + 1) % self.log_step == 0:

                wandb.log(
                    {
                        "G_LOSS_ID": vc_loss_id.item(),
                        "G_LOSS_ID_PSNET": vc_loss_id_psnt.item(),
                        "G_LOSS_CD": g_loss_cd.item(),
                        "A_LOSS_Adjust": a_loss_adjust.item(),
                        "C_LOSS_TRANS": c_loss_trans.item(),
                        "C_LOSS_RECONST": c_loss_recontruct.item(),
                        "D_LOSS": d_loss.item(),
                    }
                )
                wandb.save("autovc_org.pt")
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            if (i + 2) % self.log_step == 0:
                os.system("cls||clear")


class Config:
    def __init__(self, model_name, data_dir, num_iters):
        self.model_name = model_name
        self.data_dir = data_dir
        self.num_iters = num_iters
        self.lambda_cd = 1
        self.lambda_ad = 1
        self.lambda_cls = 0.1
        self.lambda_dis = 1
        self.n_critic = 10
        self.batch_size = 2
        self.len_crop = 176
        self.dim_neck = 44
        self.dim_emb = 256
        self.dim_pre = 512
        self.freq = 22
        self.log_step = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="traning model name")
    parser.add_argument("--data_dir", help="traning data folder")
    parser.add_argument("--save_model_name")
    parser.add_argument("--num_iters", default=1000000, help="iter time")
    args = parser.parse_args()

    config = Config(args.model_name, args.data_dir, int(args.num_iters))

    ### Init Wandb

    wandb.init(project=f'AutoVC {datetime.date.today().strftime("%b %d")}')
    wandb.run.name = args.model_name
    wandb.run.save()
    w_config = wandb.config
    w_config.len_crop = config.len_crop
    w_config.dim_neck = config.dim_neck
    w_config.dim_emb = config.dim_emb
    w_config.freq = config.freq
    w_config.batch_size = config.batch_size

    # 加速 conv，conv 的輸入 size 不會變的話開這個會比較快
    cudnn.benchmark = True
    # Data loader.
    vcc_loader = get_loader(
        config.data_dir,
        dim_neck=config.dim_neck,
        batch_size=config.batch_size,
        len_crop=config.len_crop,
    )

    solver = Solver(vcc_loader, config)
    solver.train()
    torch.save(solver.G.state_dict(), f"{args.save_model_name}.pt")
