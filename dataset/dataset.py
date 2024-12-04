import torch
import lmdb
from .audio_example import AudioExample
from random import random
from tqdm import tqdm
import numpy as np


class SimpleDataset(torch.utils.data.Dataset):
    """
    SimpleDataset is a pytorch dataset that reads from an lmdb database.
    path: str -> path to the lmdb database
    keys: list -> list of keys to retrieve from the lmdb dataset (midi, waveform)
    """

    def __init__(
        self,
        path,
        keys=['waveform', 'metadata'],
    ) -> None:
        super().__init__()

        self.env = lmdb.open(
            path,
            lock=False,
            readonly=False,
            readahead=False,
            map_async=False,
        )
        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        self.buffer_keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index=None):

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))
        out = {}

        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    print("key: ", key, " not found")
        return out


from sklearn.model_selection import train_test_split


class CombinedDataset(torch.utils.data.Dataset):
    """ 
    path_dict: dict -> dictionnary containing the path to the lmdb dataset and the frequency of each class (1.0 meaning that each dataset is sampled equally, whatever its size)
    keys: list -> list of keys to retrieve from the lmdb dataset (midi, waveform)
    config: str -> "train" or "val" or "all" to specify if the dataset is used for training, validation or to get all data
    """

    def __init__(self, path_dict, keys=["waveform"], sr=24000, config="all"):
        super().__init__()
        self.config = config

        self.sr = sr
        self.datasets = {
            k: SimpleDataset(v["path"], keys=keys)
            for k, v in path_dict.items()
        }

        if config == "train":
            self.keys = {
                k: train_test_split(v.keys, test_size=0.05, random_state=42)[0]
                for k, v in self.datasets.items()
            }

        elif config == "val":
            self.keys = {
                k: train_test_split(v.keys, test_size=0.05, random_state=42)[1]
                for k, v in self.datasets.items()
            }

        else:
            self.keys = {k: v.keys for k, v in self.datasets.items()}

        self.len = int(np.sum([len(v) for v in self.keys.values()]))

        self.weights = {
            k: path_dict[k]["freq"] * self.len / len(v)
            for k, v in self.keys.items()
        }

        self.dataset_ids = []
        self.weights_indexes = []
        self.all_keys = []
        self.all_indexes = []

        for i, k in enumerate(self.keys.keys()):
            self.dataset_ids = self.dataset_ids + [k] * len(self.keys[k])
            self.weights_indexes += [self.weights[k]] * len(self.keys[k])
            self.all_keys.extend(self.keys[k])
            self.all_indexes.extend(list(range(len(self.keys[k]))))
        self.cache = False

    def __len__(self):
        return int(self.len)

    def get_sampler(self):
        if self.config == "train":
            return torch.utils.data.WeightedRandomSampler(self.weights_indexes,
                                                          self.len,
                                                          replacement=True,
                                                          generator=None)
        elif self.config == "val":
            return torch.utils.data.WeightedRandomSampler(
                self.weights_indexes,
                self.len,
                replacement=True,
                generator=torch.Generator().manual_seed(42))

    def build_cache(self):
        print("building cache")
        self.data = {}
        for k in self.datasets.keys():
            datalist = []
            for idx in tqdm(range(len(self.keys[k]))):
                datalist.append(self.datasets[k][idx])
            self.data[k] = datalist

        self.cache = True

    def __getitem__(self, idx):
        if self.cache == False:
            dataset_id = self.dataset_ids[idx]
            data = self.datasets[dataset_id].__getitem__(
                key=self.all_keys[idx])
        else:
            dataset_id = self.dataset_ids[idx]
            data = self.data[dataset_id][self.all_indexes[idx]]

        data["label"] = dataset_id

        return data
