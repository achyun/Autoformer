import numpy as np
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
        self.metadata = self.build_metadata(config.metadata)
        self.vocoder = MelVocoder(model_name="model/static/multi_speaker")

    def build_metadata(self, metadata):
        metadata_copy = []
        for data in metadata:
            if data[0] in self.all_speaker:
                metadata_copy.append(data)
        return sorted(metadata_copy)

    def get_crop_mel(self, tmp, len_crop=176):
        if tmp.shape[0] < len_crop:
            pad_size = int(len_crop - tmp.shape[0])
            npad = [(0, 0)] * tmp.ndim
            npad[0] = (0, pad_size)
            tmp = np.pad(tmp, pad_width=npad, mode="constant", constant_values=0)
            melsp = torch.from_numpy(tmp)

        elif tmp.shape[0] == len_crop:
            melsp = torch.from_numpy(tmp)
        else:
            left = np.random.randint(0, tmp.shape[0] - len_crop)
            melsp = torch.from_numpy(tmp[left : left + len_crop, :])
        return melsp.to(self.device)

    def get_mel(self, speaker_id, sound_id):
        path_ = self.metadata[speaker_id][sound_id].replace("\\", "/")
        return self.get_crop_mel(np.load(f"{self.root}/{path_}")).unsqueeze(0)

    def get_trans_mel(self, model, source_id, target_id, sound_id=3, isAdjust=False):
        mel_source = self.get_mel(source_id, sound_id)
        emb_org = torch.from_numpy(
            np.expand_dims(self.metadata[source_id][1], axis=0)
        ).to(self.device)
        emb_trg = torch.from_numpy(
            np.expand_dims(self.metadata[target_id][1], axis=0)
        ).to(self.device)
        if isAdjust:
            mel_target = self.get_mel(target_id, sound_id)
            _, _, mel_trans, _ = model(mel_source, emb_org, emb_trg, True, mel_target)
        else:
            _, mel_trans, _ = model(mel_source, emb_org, emb_trg)
        return mel_source, mel_target, mel_trans[0, :, :]

    def get_wavs(self, mel):
        # input -> (1,80,crop_len)
        return self.vocoder.inverse(mel)
