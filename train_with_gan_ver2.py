import importlib
import time
import torch
import datetime
import torch.nn.functional as F
import wandb
import os
import argparse
from torch.backends import cudnn
from util.data_loader import get_loader
from factory.Discriminator import Discriminator
from factory.Patcher import Patcher
from factory.MetaDV import MetaDV


class Solver(object):
    """
    AutoVC + Adjust Speaker Embedding + Gan
    """

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader and All Speaker style
        self.vcc_loader = vcc_loader

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
        self.num_speaker = config.num_speaker

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.pretrained_step = config.pretrained_step
        self.pretrained_embedder_path = "model/static/metadv_vctk80.pth"

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.cosin_label = torch.ones(self.batch_size).to(self.device)
        self.log_step = config.log_step
        self.log_keys = [
            "VC/loss_id",
            "VC/loss_id_psnt",
            "VC/loss_id_patcher",
            "VC/loss_cd_g",
            "VC/loss_cd_p",
            "A/loss_adjust_reconst",
            "A/loss_adjust",
            "C/loss_trans_g",
            "C/loss_trans_p",
            "D/loss_g",
            "D/loss_p",
        ]

        # Build the model.
        self.build_model()

    def build_model(self):

        self.VC = getattr(
            importlib.import_module(f"factory.{self.model_name}"), self.model_name
        )(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.vc_optimizer = torch.optim.Adam(self.VC.parameters(), 0.0001)
        self.VC.to(self.device)

        # 補丁，拿 VC 生成的聲音跟欲生成目標的特徵再去生成一次
        self.P = Patcher()
        self.p_optimizer = torch.optim.Adam(self.P.parameters(), 0.0001)
        self.P.to(self.device)

        # 拿來分類轉換後聲音的，算 speaker embedding 之間的 cos-similiarty 或 MSE
        self.C = MetaDV(self.num_speaker)
        print(f"Load Pretrained Embedder from --- {self.pretrained_embedder_path}")
        self.C.load_state_dict(
            torch.load(self.pretrained_embedder_path, map_location=self.device)
        )
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), 0.0001)
        self.C.to(self.device)

        # 判斷聲音真假
        self.D = Discriminator()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), 0.0001)
        self.D.to(self.device)

    def reset_grad(self):
        self.vc_optimizer.zero_grad()
        self.p_optimizer.zero_grad()
        self.c_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def discriminator_loss(self, real, fake):
        real_loss = torch.nn.BCELoss()(real, torch.ones_like(real))
        fake_loss = torch.nn.BCELoss()(fake, torch.zeros_like(fake))
        return real_loss + fake_loss

    def get_data(self):
        try:
            x, style = next(data_iter)
        except:
            data_iter = iter(self.vcc_loader)
            x, style = next(data_iter)
        x = x.to(self.device)
        style = style.to(self.device)
        return x, style

    def print_log(self, loss, i, start_time):
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
        for tag in self.log_keys:
            log += ", {}: {:.4f}".format(tag, loss[tag])
        print(log)

    def train(self):

        print("Start training...")
        start_time = time.time()
        for i in range(self.num_iters):
            x_source, org_style = self.get_data()
            self.VC = self.VC.train()
            self.C = self.C.train()
            self.D = self.D.train()

            org_style_adjust, x_identic, source_mel_g, code_real = self.VC(
                x_source, org_style, org_style
            )
            source_mel_p = self.P(source_mel_g.squeeze(), org_style_adjust)

            # Identity mapping loss
            vc_loss_id = F.mse_loss(x_source, x_identic.squeeze())
            vc_loss_id_psnt = F.mse_loss(x_source, source_mel_g.squeeze())
            vc_loss_id_patcher = F.mse_loss(x_source, source_mel_p)

            code_reconst_g = self.VC(source_mel_g, org_style, None)
            code_reconst_p = self.P(source_mel_p, org_style, None)
            # Code semantic loss.
            vc_loss_cd_g = F.l1_loss(code_real, code_reconst_g)
            vc_loss_cd_p = F.l1_loss(code_real, code_reconst_p)

            # Adjust Speaker embedding loss
            a_loss_adjust_reconst = F.l1_loss(org_style_adjust, org_style)
            autovc_adjust_loss = (
                vc_loss_id
                + vc_loss_id_psnt
                + vc_loss_id_patcher
                + self.lambda_cd * (vc_loss_cd_g + vc_loss_cd_p)
                + a_loss_adjust_reconst
            )

            if (i + 1) % self.n_critic == 0 and (i + 1) > self.pretrained_step:
                x_target, target_style = self.get_data()
                org_style_adjust, _, target_mel_g, _ = self.VC(
                    x_source, org_style, target_style, True, x_target
                )
                target_mel_p = self.P(target_mel_g, target_style)

                # 抽出 style
                _, trans_style_g = self.C(target_mel_g.squeeze())
                _, trans_style_p = self.C(target_mel_p.squeeze())

                # x_target 是 Real Data
                real_prob = self.D(x_target)
                fake_prob_g = self.D(target_mel_g.squeeze())
                fake_prob_p = self.D(target_mel_p.squeeze())

                # Cosine Embedding Loss 接近 0.1 時就代表轉換的不錯了
                # Adjust Speaker embedding loss
                a_loss_adjust = F.cosine_embedding_loss(
                    org_style_adjust, org_style, self.cosin_label
                )
                # Classifier loss
                c_loss_trans_g = F.cosine_embedding_loss(
                    trans_style_g, target_style, self.cosin_label
                )
                c_loss_trans_p = F.cosine_embedding_loss(
                    trans_style_p, target_style, self.cosin_label
                )

                # Discriminator loss
                d_loss_g = self.discriminator_loss(real_prob, fake_prob_g)
                d_loss_p = self.discriminator_loss(real_prob, fake_prob_p)
                autovc_gan_loss = (
                    autovc_adjust_loss
                    + self.lambda_ad * a_loss_adjust
                    + self.lambda_cls * (c_loss_trans_g + c_loss_trans_p)
                    + self.lambda_dis * (d_loss_g + d_loss_p)
                )
                self.reset_grad()
                autovc_gan_loss.backward()
                self.vc_optimizer.step()
                self.p_optimizer.step()
                self.c_optimizer.step()
                self.d_optimizer.step()

            else:
                self.reset_grad()
                autovc_adjust_loss.backward()
                self.vc_optimizer.step()
                self.p_optimizer.step()

            if (i + 1) > self.pretrained_step and (i + 1) % self.log_step == 0:
                loss = {}
                loss["VC/loss_id"] = vc_loss_id.item()
                loss["VC/loss_id_psnt"] = vc_loss_id_psnt.item()
                loss["VC/loss_id_patcher"] = vc_loss_id_patcher.item()
                loss["VC/loss_cd_g"] = vc_loss_cd_g.item()
                loss["VC/loss_cd_p"] = vc_loss_cd_p.item()
                loss["A/loss_adjust_reconst"] = a_loss_adjust_reconst.item()
                loss["A/loss_adjust"] = a_loss_adjust.item()
                loss["C/loss_trans_g"] = c_loss_trans_g.item()
                loss["C/loss_trans_p"] = c_loss_trans_p.item()
                loss["D/loss_g"] = d_loss_g.item()
                loss["D/loss_p"] = d_loss_p.item()
                wandb.log(
                    {
                        "VC_LOSS_ID": vc_loss_id.item(),
                        "VC_LOSS_ID_PSNET": vc_loss_id_psnt.item(),
                        "VC_LOSS_ID_PATCHER": vc_loss_id_patcher.item(),
                        "VC_LOSS_CD_G": vc_loss_cd_g.item(),
                        "VC_LOSS_CD_P": vc_loss_cd_p.item(),
                        "A/LOSS_ADJUST_RECONST": a_loss_adjust_reconst.item(),
                        "A/LOSS_ADJUST": a_loss_adjust.item(),
                        "C/LOSS_TRANS_G": c_loss_trans_g.item(),
                        "C/LOSS_TRANS_P": c_loss_trans_p.item(),
                        "D/LOSS_G": d_loss_g.item(),
                        "D/LOSS_P": d_loss_p.item(),
                    }
                )
                self.print_log(loss, i, start_time)

            elif (i + 1) % self.log_step == 0:
                loss = {}
                loss["VC/loss_id"] = vc_loss_id.item()
                loss["VC/loss_id_psnt"] = vc_loss_id_psnt.item()
                loss["VC/loss_id_patcher"] = vc_loss_id_patcher.item()
                loss["VC/loss_cd_g"] = vc_loss_cd_g.item()
                loss["VC/loss_cd_p"] = vc_loss_cd_p.item()
                loss["A/loss_adjust_reconst"] = a_loss_adjust_reconst.item()
                loss["A/loss_adjust"] = 0.0
                loss["C/loss_trans_g"] = 0.0
                loss["C/loss_trans_p"] = 0.0
                loss["D/loss_g"] = 0.0
                loss["D/loss_p"] = 0.0
                wandb.log(
                    {
                        "VC_LOSS_ID": vc_loss_id.item(),
                        "VC_LOSS_ID_PSNET": vc_loss_id_psnt.item(),
                        "VC_LOSS_ID_PATCHER": vc_loss_id_patcher.item(),
                        "VC_LOSS_CD_G": vc_loss_cd_g.item(),
                        "VC_LOSS_CD_P": vc_loss_cd_p.item(),
                        "A/LOSS_ADJUST_RECONST": a_loss_adjust_reconst.item(),
                    }
                )
                self.print_log(loss, i, start_time)

            if (i + 2) % self.log_step == 0:
                os.system("cls||clear")


class Config:
    def __init__(self, model_name, data_dir, num_iters):
        self.model_name = model_name
        self.data_dir = data_dir
        self.num_iters = num_iters
        self.pretrained_step = int(num_iters / 10)
        self.num_speaker = 80
        self.lambda_cd = 1
        self.lambda_ad = 0.1
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

    print(" --- Use Config  ---")
    print(f" Lambda cd ---  {config.lambda_cd}")
    print(f" Lambda ad --- {config.lambda_ad}")
    print(f" Lambda cls --- {config.lambda_cls}")
    print(f" Lambda dis--- {config.lambda_dis}")
    print(f" N critic --- {config.n_critic }")
    print(f" VC Pretrained step  --- {config.pretrained_step  }")
    print(" ----------------------")

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
    torch.save(solver.VC.state_dict(), f"{args.save_model_name}.pt")