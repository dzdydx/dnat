# -*- coding: utf-8 -*-
# @Time    : 2022-5-17 22:25:34
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn

import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
from pathlib import Path
import random

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['label']] = row['idx']
            line_count += 1
    return index_lookup

class WavDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, sample_rate=32000):
        metadata = json.load(Path(dataset_json_file).open())
        self.data_dir = Path(metadata['root_dir'])
        self.data = metadata['data']

        self.__dict__.update(audio_conf)
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)

    def __getitem__(self, index):
        """
        returns: a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        datum = self.data[index]
        filename = self.data_dir / datum['filename']

        waveform, sr = torchaudio.load(filename)
        if sr != self.sample_rate:
            resampler = torchaudio.functional.resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)
        
        label_indices = np.zeros(self.label_num)
        for label_str in datum['labels']:
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        return waveform, label_indices, datum['filename']
        
    def __len__(self):
        return len(self.data)