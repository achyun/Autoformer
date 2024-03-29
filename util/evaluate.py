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
        self.max_uttr_idx = config.max_uttr_idx
        self.erroment_num = config.erroment_num
        self.len_crop = config.len_crop
        self.device = config.device
        self.all_speaker = config.all_speaker
        self.embedder = config.embedder
        self.enrroment_idx = []
        self.remain_idx = np.arange(2, config.max_uttr_idx)
        self.metadata = self.build_metadata(config.metadata)

        self.vocoder = MelVocoder(model_name="model/static/multi_speaker")
        if self.embedder != None:
            print("Detect Judge ! generate all Real Data d-vector")
            self.all_dv = self.generate_real_dv()

    def build_metadata(self, metadata):
        metadata_copy = []
        for data in metadata:
            if str(data[0]) in self.all_speaker:
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
        isAdain: bool,
        isPlay=False,
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
        elif isAdain:
            _, feature = model(mel_source, emb_org, None, None)
            _, mel_trans, _ = model(mel_source, emb_org, emb_trg, feature)
        else:
            _, mel_trans, _ = model(mel_source, emb_org, emb_trg)

        mel_trans = mel_trans.squeeze(1)

        if isPlay:
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
        for enrroment_idx in self.enrroment_idx:
            mel, _ = self.get_mel(speaker_id, enrroment_idx)
            _dv += self.embedder(mel)[1].detach().cpu()
        _dv = _dv / (self.erroment_num)
        return _dv.to(self.device)

    def generate_real_dv(self):
        all_dv = []
        for _ in range(self.erroment_num):
            # random pick 16 utterence for enrroment
            enrroment_idx = random.choice(self.remain_idx)
            self.remain_idx = np.delete(
                self.remain_idx, np.where(self.remain_idx == enrroment_idx)
            )
            self.enrroment_idx.append(enrroment_idx)

        for i, speaker in enumerate(self.all_speaker):
            print(f"Processing --- ID:{i} Speaker:{speaker} ---")
            all_dv.append(self.get_dv(i))
        return all_dv

    def get_real_data_cos(self):
        cos_result = np.zeros((self.num_speaker, self.num_speaker))
        var_result = np.zeros((self.num_speaker, self.num_speaker))
        for i, speaker in enumerate(self.all_speaker):
            print(f"Processing --- ID:{i} Speaker:{speaker} ---")
            tmp_style = []
            # Generate 52 embed first
            for sound_id in self.remain_idx:
                mel, _ = self.get_mel(i, sound_id)
                tmp_style.append(self.embedder(mel)[1].detach().cpu())

            # Compare with all dv
            for j, data in enumerate(self.all_dv):
                tmp_cos = []
                for _style in tmp_style:
                    dv = data.detach().cpu()
                    tmp_cos.append(
                        torch.clamp(
                            nn.functional.cosine_similarity(
                                dv, _style, dim=1, eps=1e-8
                            ),
                            min=0.0,
                        )
                        .numpy()[0]
                        .astype(np.float32)
                    )
                cos_result[i][j] = np.array(tmp_cos).mean()
                var_result[i][j] = np.array(tmp_cos).std()
        return cos_result, var_result

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

    def generate_result(self, models: list, sound_id=2, isAdjust=False, isAdain=False):
        cos_result = []

        for _ in range(len(models)):
            cos_result.append(
                np.zeros((self.num_speaker, self.num_speaker, self.num_speaker))
            )

        for source_id, data in enumerate(self.metadata):
            sp_s = data[0]
            print(f"Now Processing --- {sp_s}")
            for target_id, data in enumerate(self.metadata):
                sp_o = data[0]
                _dv_result = []
                if source_id == target_id:
                    print(f"Reconstruct ---- {sp_s} to {sp_o}")
                else:
                    print(f"Trans --- {sp_s} to {sp_o}")

                for model_id, model in enumerate(models):
                    _, _, trans_mel = self.get_trans_mel(
                        model, source_id, target_id, sound_id, isAdjust, isAdain
                    )
                    trans_style = self.embedder(trans_mel)[1]
                    _dv_result.append(trans_style)

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
                        cos_result[model_id][source_id][target_id][k] = cos
        return cos_result
