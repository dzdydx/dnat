# -*- coding: utf-8 -*-
# @Time    : 2022-5-10 11:09:50
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn
# Modified from https://github.com/YuanGongND/ast

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

class AudioTaggingDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        metadata = json.load(Path(dataset_json_file).open())
        self.data_dir = Path(metadata['root_dir'])
        self.data = metadata['data']

        self.__dict__.update(audio_conf)

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        else:
            # mixup
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, sample_frequency=self.sample_rate, window_type='hanning',
                                          num_mel_bins=self.num_mel_bins)
        # fbank.shape = [998, 128] for audioset, [498, 128] for esc-50

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        datum = self.data[index]
        file1 = self.data_dir / datum['filename']
        if random.random() < self.mixup:
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            file2 = self.data_dir / mix_datum['filename']
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(file1, file2)
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels']:
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels']:
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(file1)
            for label_str in datum['labels']:
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freq_mask != 0:
            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask)
            fbank = freqm(fbank)
        if self.time_mask != 0:
            timem = torchaudio.transforms.TimeMasking(self.time_mask)
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # if self.noise == True:
        #     fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        #     fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices, datum['filename']

    def __len__(self):
        return len(self.data)