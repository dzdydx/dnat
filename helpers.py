# -*- coding: utf-8 -*-
# @Time    : 2022-5-10 16:02:05
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn

# this is a sample code of how to get normalization stats for input spectrogram
import torch
import numpy as np
from tqdm import tqdm
from data.audio_tagging import AudioTaggingDataset
audio_conf = {'sample_rate': 44100, 'num_mel_bins': 128, 'time_frames': 498, 'skip_norm': True, 'norm_mean':-5.025409, 'norm_std': 5.4591165, 'freq_mask': 24, 'time_mask': 96, 'mixup': 0}

json_file = "tasks/esc50/metadata/esc50_full.json"
label_csv = "tasks/esc50/metadata/label_vocabulary.csv"

train_loader = torch.utils.data.DataLoader(
    AudioTaggingDataset(dataset_json_file=json_file, label_csv=label_csv,
                                audio_conf=audio_conf), batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
mean=[]
std=[]
for epoch in range(10):
    for i, (audio_input, labels) in enumerate(tqdm(train_loader)):
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
    print(f"Epoch {epoch} -- norm_mean: {np.mean(mean)}, norm_std: {np.mean(std)}")
print(f"norm_mean: {np.mean(mean)}, norm_std: {np.mean(std)}")