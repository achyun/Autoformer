from torch.utils import data
import torch
import numpy as np
import pickle
import os

from multiprocessing import Process, Manager


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.step = 10

        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))

        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta) * [None])
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(
                target=self.load_data, args=(meta[i : i + self.step], dataset, i)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)

        print("Finished loading the style dataset...")

    def load_data(self, submeta, dataset, idx_offset):
        for k, sbmt in enumerate(submeta):
            uttrs = len(sbmt) * [None]
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
            dataset[idx_offset + k] = uttrs

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset
        list_uttrs = dataset[index]
        return list_uttrs[1]

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


def get_style_loader(root_dir, batch_size=2, num_workers=0):
    """Build and return a data loader."""

    dataset = Utterances(root_dir)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 44))
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    return data_loader
