# -*- coding: utf-8 -*-
# @Time    : 2022-5-17 22:25:34
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn

# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .config import get_dataset_conf
from .audio_tagging_dataset import AudioTaggingDataset
from .wav_dataset import WavDataset

class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 dataset_type='audio_tagging_dataset',
                 **kwargs):
        super().__init__()

        # Dataset paths
        self.load_data_module(dataset_type)

        self.train_json=kwargs.get("train_json")
        self.val_json=kwargs.get("val_json")
        self.test_json=kwargs.get("test_json")
        self.label_csv=kwargs.get("label_csv")

        # Audio configs
        self.train_audio_conf, self.val_audio_conf = get_dataset_conf(dataset, **kwargs)

        self.num_workers = num_workers
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.sample_rate = kwargs['sample_rate']

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.data_module(self.train_json, self.train_audio_conf, self.label_csv, self.sample_rate)
            if self.val_json is not None:
                self.valset = self.data_module(self.val_json, self.val_audio_conf, self.label_csv, self.sample_rate)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.data_module(self.test_json, self.val_audio_conf, self.label_csv, self.sample_rate)

    #     # If you need to balance your data using Pytorch Sampler,
    #     # please uncomment the following lines.
    
    #     with open('./data/ref/samples_weight.pkl', 'rb') as f:
    #         self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self, dataset_type):
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in dataset_type.split('_')])
        try:
            if dataset_type == "audio_tagging_dataset":
                self.data_module = AudioTaggingDataset
            elif dataset_type == "wav_dataset":
                self.data_module = WavDataset
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')
