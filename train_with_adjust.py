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
    AutoVC 加上 Adjust Speaker Embedding
    """

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.lambda_ad = config.lambda_ad
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.model_name = config.model_name
        self.use_pretrained_weight = config.use_pretrained_weight
        self.pretrained_weight_path = config.pretrained_weight_path
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
        self.G = getattr(
            importlib.import_module(f"factory.{self.model_name}"), self.model_name
        )(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        if self.use_pretrained_weight == True:
            print(f"Load Pre-trained Weight --- {self.pretrained_weight_path}")
            self.G.load_state_dict(
                torch.load(self.pretrained_weight_path, map_location=self.device)
            )
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        self.G.to(self.device)

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def train(self):

        data_loader = self.vcc_loader
        keys = ["G/loss_id", "G/loss_id_psnt", "G/loss_cd", "A/loss_adjust"]

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

            self.G = self.G.train()

            # Identity mapping loss
            emb_adjust, x_identic, x_identic_psnt, code_real = self.G(
                x_real, emb_org, emb_org
            )
            g_loss_id = F.mse_loss(x_real, x_identic.squeeze())
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())

            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            # Adjust Speaker embedding loss
            a_loss_adjust = F.l1_loss(emb_adjust, emb_org)

            # Backward and optimize.
            g_loss = (
                g_loss_id
                + g_loss_id_psnt
                + self.lambda_cd * g_loss_cd
                + self.lambda_ad * a_loss_adjust
            )
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss["G/loss_id"] = g_loss_id.item()
            loss["G/loss_id_psnt"] = g_loss_id_psnt.item()
            loss["G/loss_cd"] = g_loss_cd.item()
            loss["A/loss_adjust"] = a_loss_adjust.item()

            # =================================================================================== #
            #                               4. Print Traning Info                                 #
            # =================================================================================== #

            if (i + 1) % self.log_step == 0:

                wandb.log(
                    {
                        "G_LOSS_ID": g_loss_id.item(),
                        "G_LOSS_ID_PSNET": g_loss_id_psnt.item(),
                        "G_LOSS_CD": g_loss_cd.item(),
                        "A_LOSS_Adjust": a_loss_adjust.item(),
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
    def __init__(
        self,
        model_name,
        data_dir,
        num_iters,
        use_pretrained_weight,
        pretrained_weight_path,
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.num_iters = num_iters
        self.use_pretrained_weight = use_pretrained_weight
        self.pretrained_weight_path = pretrained_weight_path
        self.batch_size = 2
        self.len_crop = 176
        self.lambda_cd = 1
        self.lambda_ad = 1
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
    parser.add_argument("--use_pretrained_weight", default=0)
    parser.add_argument("--pretrained_weight_path", default="")
    parser.add_argument("--num_iters", default=1000000, help="iter time")
    args = parser.parse_args()

    config = Config(
        args.model_name,
        args.data_dir,
        int(args.num_iters),
        bool(args.use_pretrained_weight),
        args.pretrained_weight_path,
    )

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

    solver.train()
    torch.save(solver.G.state_dict(), f"{args.save_model_name}.pt")
