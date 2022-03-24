import os
import glob
import random

import torch
import torchaudio

from utils import load_sets, save_sets

random.seed(0)

class AudioMNISTDataset:
    def __init__(self, path, transforms_train=None, transforms_test=None, load_all=False, **args):
        if path is None:
            raise ValueError('Expected path to be a directory containing WAV recordings')

        self.path = path
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test
        self.load_all = load_all
        self.args = args
        self.all_files = glob.glob(os.path.join(self.path, '*/*.wav'))
        random.shuffle(self.all_files)
        _, self.sample_rate = torchaudio.load(self.all_files[0])

    def full(self):
        return AudioMNIST(self.all_files, self.transforms_train, self.load_all, **self.args)

    def train_test_split(self, test_size=0.2):
        assert 0. < test_size < 1.
        name = self.path+"/test_" + str(test_size) + ".txt"
        exits = os.path.exists(name)
        if exits:
            train_files, test_files = load_sets(self.path, test_size)
        else:
            tot = len(self.all_files)
            pivot = int(tot*(1-test_size))
            train_files = self.all_files[:pivot]
            test_files = self.all_files[pivot:]
            save_sets(self.path, train_files, test_files, test_size)

        train_set = AudioMNIST(train_files, self.transforms_train, self.load_all, **self.args)
        test_set = AudioMNIST(test_files, self.transforms_test, self.load_all, **self.args)
        return train_set, test_set

class AudioMNIST(torch.utils.data.Dataset):

    def __init__(self, files, transforms=None, load_all=False, **args):
        super().__init__()
        self.files = files
        self.transforms = transforms
        self.args = args

        get_audio = lambda file: torchaudio.load(file, **self.args)[0]
        get_label = lambda file: int(os.path.basename(file)[0])

        if load_all:
            self.recordings, self.labels = [], []
            for file in self.files:
                self.recordings.append(get_audio(file))
                self.labels.append(get_label(file))

            def _load(self, index):
                return self.recordings[index], self.labels[index]
        else:
            def _load(self, index):
                file = self.files[index]
                return file, get_audio(file), get_label(file)

        setattr(self.__class__, '_load', _load)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Fetch the audio and corresponding label
        file, x, y = self._load(index)
        x = x.flatten()

        # Transform data if a transformation is given
        if self.transforms is not None:
            x = self.transforms(x)

        return file, x, y


if __name__ == '__main__':
    ds = AudioMNISTDataset("./dataset/")
    train, test = ds.train_test_split()
    print(train[0])






















