import numpy as np
import random
import torch.nn as nn
from factory import *
from melgan.interface import *


class Evaluator:
    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.root = config.root
        self.num_speaker = config.num_speaker
        self.batch_size = config.batch_size
        self.erroment_uttr_idx = config.erroment_uttr_idx
        self.max_uttr_idx = config.max_uttr_idx
        self.len_crop = config.len_crop
        self.device = config.device
        self.all_speaker = config.all_speaker
        self.judge = config.judge
        self.metadata = self.build_metadata(config.metadata)
        self.vocoder = MelVocoder(model_name="model/static/multi_speaker")
        if self.judge != None:
            print("Detect Judge ! generate all Real Data d-vector")
            self.all_dv = self.generate_real_dv()

    def build_metadata(self, metadata):
        metadata_copy = []
        for data in metadata:
            if data[0] in self.all_speaker:
                metadata_copy.append(data)
        return sorted(metadata_copy)

    def crop_mel(self, tmp):
        pad_size = 0
        if tmp.shape[0] < self.len_crop:
            pad_size = int(self.len_crop - tmp.shape[0])
            npad = [(0, 0)] * tmp.ndim
            npad[0] = (0, pad_size)
            tmp = np.pad(tmp, pad_width=npad, mode="constant", constant_values=0)
            melsp = torch.from_numpy(tmp)

        elif tmp.shape[0] == self.len_crop:
            melsp = torch.from_numpy(tmp)
        else:
            left = np.random.randint(0, tmp.shape[0] - self.len_crop)
            melsp = torch.from_numpy(tmp[left : left + self.len_crop, :])
        return melsp.unsqueeze(0).to(self.device), pad_size

    def get_mel(self, speaker_id, sound_id):
        path_ = self.metadata[speaker_id][sound_id].replace("\\", "/")
        return self.crop_mel(np.load(f"{self.root}/{path_}"))

    def get_trans_mel(
        self,
        model: nn.Module,
        source_id: int,
        target_id: int,
        sound_id: int,
        isAdjust: bool,
    ):
        mel_source, pad_size_source = self.get_mel(source_id, sound_id)
        mel_target, pad_size_target = self.get_mel(target_id, sound_id)

        emb_org = (
            torch.from_numpy(self.metadata[source_id][1]).unsqueeze(0).to(self.device)
        )

        emb_trg = (
            torch.from_numpy(self.metadata[target_id][1]).unsqueeze(0).to(self.device)
        )

        if isAdjust:
            _, _, mel_trans, _ = model(mel_source, emb_org, emb_trg, True, mel_target)
        else:
            _, mel_trans, _ = model(mel_source, emb_org, emb_trg)
        mel_trans = mel_trans.squeeze(1)

        if pad_size_source > 0:
            mel_source = mel_source[:, : (self.len_crop - pad_size_source), :]
            mel_trans = mel_trans[:, : (self.len_crop - pad_size_source), :]
        if pad_size_target > 0:
            mel_target = mel_target[:, : (self.len_crop - pad_size_target), :]

        return mel_source, mel_target, mel_trans

    def get_wavs(self, mel):
        # input -> (1,80,crop_len)
        return self.vocoder.inverse(mel)

    def get_dv(self, speaker_id):
        _dv = torch.zeros((1, 256))
        for _ in range(self.erroment_uttr_idx):
            mel, _ = self.get_mel(speaker_id, random.randint(2, self.max_uttr_idx))
            _dv += self.judge(mel)[1].detach().cpu()
        _dv = _dv / (self.erroment_uttr_idx)
        return _dv.to(self.device)

    def generate_real_dv(self):
        all_dv = []
        for i, speaker in enumerate(self.all_speaker):
            print(f"Processing --- ID:{i} Speaker:{speaker} ---")
            all_dv.append(self.get_dv(i))
        return all_dv

    def get_real_data_result(self):
        cos_result = np.zeros((self.num_speaker, self.num_speaker))
        for i, speaker in enumerate(self.all_speaker):
            print(f"Processing --- ID:{i} Speaker:{speaker} ---")
            mel, _ = self.get_mel(i, random.randint(2, self.max_uttr_idx))
            _dv = self.judge(mel)[1].detach().cpu()
            for j, data in enumerate(self.all_dv):
                dv = data.detach().cpu()
                cos = (
                    torch.clamp(
                        nn.functional.cosine_similarity(_dv, dv, dim=1, eps=1e-8),
                        min=0.0,
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                    .astype(np.float32)
                )
                cos_result[i][j] = cos
        return cos_result

    def get_cos_rc(self, cos_res):
        rc = 0.0
        rc_ = 0.0
        N = len(cos_res)
        for i in range(N):
            total = sum(cos_res[i][i])
            others = (total - cos_res[i][i][i]) / (N - 1)
            rc_ += others
            rc += cos_res[i][i][i]
        return rc / N, rc_ / N

    def get_cos_trans(self, cos_res):
        trans = []
        trans_ = []
        N = len(cos_res)
        for i in range(N):
            trans_cos = np.sum(np.diagonal(cos_res[i])) - cos_res[i][i][i]
            trans.append(trans_cos / (N - 1))
            trans_.append(
                (np.sum(cos_res[i]) - np.sum(np.diagonal(cos_res[i]))) / ((N - 1) * N)
            )

        return sum(trans) / len(trans), sum(trans_) / len(trans_)

    def generate_result(self, models: list, isAdjust=False):
        cos_reslut = []
        for _ in range(len(models)):
            cos_reslut.append(
                np.zeros((self.num_speaker, self.num_speaker, self.num_speaker))
            )
        for source_id, data in enumerate(self.metadata):
            sp_s = data[0]
            sound_id = random.randint(2, self.max_uttr_idx)
            print(f"Now Processing --- {sp_s}")
            for target_id, data in enumerate(self.metadata):
                sp_o = data[0]
                _dv_result = []
                for model in models:
                    _, _, trans_mel = self.get_trans_mel(
                        model, source_id, target_id, sound_id, isAdjust
                    )
                    _dv_result.append(self.judge(trans_mel)[1])
                if source_id == target_id:
                    print(f"Reconstruct ---- {sp_s} to {sp_o}")
                else:
                    print(f"Trans --- {sp_s} to {sp_o}")
                for k, emb in enumerate(self.all_dv):
                    for model_id, _dv in enumerate(_dv_result):
                        cos = (
                            torch.clamp(
                                nn.functional.cosine_similarity(
                                    _dv, emb, dim=1, eps=1e-8
                                ),
                                min=0.0,
                            )
                            .detach()
                            .cpu()
                            .numpy()[0]
                            .astype(np.float32)
                        )
                        cos_reslut[model_id][source_id][target_id][k] = cos
        return cos_reslut
