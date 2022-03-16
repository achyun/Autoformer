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


class Solver(object):
    """
    最原始的 AutoVC 訓練方法
    """

    def __init__(self, vcc_loader, config):
        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
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

    def reset_grad(self):
        self.vc_optimizer.zero_grad()

    def train(self):

        data_loader = self.vcc_loader
        keys = ["VC/loss_id", "VC/loss_id_psnt", "VC/loss_cd"]

        print("Start training...")
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)

            x_real = x_real.to(self.device)
            emb_org = emb_org.to(self.device)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            self.VC = self.VC.train()

            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.VC(x_real, emb_org, emb_org)
            vc_loss_id = F.mse_loss(x_real, x_identic.squeeze())
            vc_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())

            # Code semantic loss.
            code_reconst = self.VC(x_identic_psnt, emb_org, None)
            vc_loss_cd = F.l1_loss(code_real, code_reconst)

            # Backward and optimize.
            vc_loss = vc_loss_id + vc_loss_id_psnt + self.lambda_cd * vc_loss_cd
            self.reset_grad()
            vc_loss.backward()
            self.vc_optimizer.step()

            # Logging.
            loss = {}
            loss["VC/loss_id"] = vc_loss_id.item()
            loss["VC/loss_id_psnt"] = vc_loss_id_psnt.item()
            loss["VC/loss_cd"] = vc_loss_cd.item()

            # =================================================================================== #
            #                               4. Print Traning Info                                 #
            # =================================================================================== #

            if (i + 1) % self.log_step == 0:
                """
                wandb.log(
                    {
                        "VC_LOSS_ID": vc_loss_id.item(),
                        "VC_LOSS_ID_PSNET": vc_loss_id_psnt.item(),
                        "VC_LOSS_CD": vc_loss_cd.item(),
                    }
                )
                wandb.save("autovc_org.pt")
                """
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
        self.batch_size = 2
        self.len_crop = 176
        self.lambda_cd = 1
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
    """
    wandb.init(project=f'AutoVC {datetime.date.today().strftime("%b %d")}')
    wandb.run.name = args.model_name
    wandb.run.save()
    w_config = wandb.config
    w_config.len_crop = config.len_crop
    w_config.dim_neck = config.dim_neck
    w_config.dim_emb = config.dim_emb
    w_config.freq = config.freq
    w_config.batch_size = config.batch_size
    """
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
