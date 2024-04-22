import json
import os
import random
from collections import defaultdict

import torch
import torchaudio
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms as T


class SENDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        img_path: str,
        audio_path: str,
        input_size=299,
        n_fft=512,
        n_mels=40,
        win_length=250,
        hop_length=100,
    ):
        super().__init__()
        data = json.load(open(json_file, "r", encoding="utf-8"))["data"]
        walker = [
            [
                dict(
                    label=datum["class"],
                    img=img_path + os.sep + datum["img"],
                    audio=audio_path + os.sep + wav,
                )
                for wav in datum["wav"]
            ]
            for datum in data
        ]
        # check exits
        self.walker = [j for i in walker for j in i]
        subset = json_file.rsplit(os.sep, 1)[-1].split("_", 1)[0]

        # self.img_transform = {
        #     'train': T.Compose(
        #         [
        #             T.ToTensor(),
        #             T.RandomRotation(degrees=(0, 180)),
        #             T.RandomHorizontalFlip(p=0.5),
        #             T.RandomVerticalFlip(p=0.5),
        #             T.Resize((input_size, input_size)),
        #             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ]
        #     ),
        #     'test': T.Compose(
        #         [
        #             T.ToTensor(),
        #             T.Resize((input_size, input_size)),
        #             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ]
        #     )
        # }[subset]

        # Augmentation for image
        

        self.img_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((input_size, input_size)),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        sample_rate = 16000  # default value
        self.audio_transform = MelSpectrogram(
            sample_rate, n_fft, win_length, hop_length, n_mels=n_mels
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        img = Image.open(item["img"])
        img = self.img_transform(img)

        wav, sr = torchaudio.load(item["audio"])
        mel_spec = self.audio_transform(wav)
        mel_spec = mel_spec.squeeze().permute(1, 0)  # (len, n_mels)

        return img, mel_spec, mel_spec.size(0), item["label"]

# sed
class SEDDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        audio_path: str,
        input_size=299,
        n_fft=512,
        n_mels=40,
        win_length=250,
        hop_length=100,
    ):
        super().__init__()
        data = json.load(open(json_file, "r", encoding="utf-8"))["data"]
        walker = [
            [
                dict(
                    label=datum["class"],
                    audio=audio_path + os.sep + wav,
                )
                for wav in datum["wav"]
            ]
            for datum in data
        ]
        # check exits
        self.walker = [j for i in walker for j in i]
 

        sample_rate = 16000  # default value
        self.audio_transform = MelSpectrogram(
            sample_rate, n_fft, win_length, hop_length, n_mels=n_mels
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]


        wav, sr = torchaudio.load(item["audio"])
        mel_spec = self.audio_transform(wav)
        mel_spec = mel_spec.squeeze().permute(1, 0)  # (len, n_mels)

        return mel_spec, mel_spec.size(0), item["label"]

class RDGDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        img_path: str,
        audio_path: str,
        input_size=299,
        n_fft=512,
        n_mels=40,
        win_length=250,
        hop_length=100,
    ):
        super().__init__()
        data = json.load(open(json_file, "r", encoding="utf-8"))["data"]
        walker = [
            [
                dict(
                    label=datum["class"],
                    img=img_path + os.sep + datum["img"],
                    audio=audio_path + os.sep + wav,
                )
                for wav in datum["wav"]
            ]
            for datum in data
        ]
        # check exits
        self.walker = [j for i in walker for j in i]

        self.data_class = defaultdict(list)
        for data in self.walker:
            self.data_class[data.get("label")].append(data)

        subset = json_file.rsplit(os.sep, 1)[-1].split("_", 1)[0]

        # self.img_transform = {
        #     'train': T.Compose(
        #         [
        #             T.ToTensor(),
        #             T.RandomRotation(degrees=(0, 180)),
        #             T.RandomHorizontalFlip(p=0.5),
        #             T.RandomVerticalFlip(p=0.5),
        #             T.Resize((input_size, input_size)),
        #             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ]
        #     ),
        #     'test': T.Compose(
        #         [
        #             T.ToTensor(),
        #             T.Resize((input_size, input_size)),
        #             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ]
        #     )
        # }[subset]

        self.img_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((input_size, input_size)),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        sample_rate = 16000  # default value
        self.audio_transform = MelSpectrogram(
            sample_rate, n_fft, win_length, hop_length, n_mels=n_mels
        )

    def __len__(self):
        return len(self.walker)

    def __get_random_same_class__(self, label):
        data = self.data_class[label]
        l, h = 1, 102
        return data[random.randint(l, h)]

    def __get_random_diff_class__(self, diff_label):
        l, h = 1, 102
        label = random.randint(l, h)
        while label == diff_label:
            label = random.randint(l, h)
        return self.__get_random_same_class__(label)

    def __getitem__(self, index):
        item = self.walker[index]
        label = item["label"]

        real_img = Image.open(item["img"])
        similar_img = Image.open(self.__get_random_same_class__(label)["img"])
        wrong_img = Image.open(self.__get_random_diff_class__(label)["img"])

        real_img = self.img_transform(real_img)
        similar_img = self.img_transform(similar_img)
        wrong_img = self.img_transform(wrong_img)

        wav, sr = torchaudio.load(item["audio"])
        mel_spec = self.audio_transform(wav)
        mel_spec = mel_spec.squeeze().permute(1, 0)  # (len, n_mels)

        return real_img, similar_img, wrong_img, mel_spec, mel_spec.size(0), (item['audio'], sr)