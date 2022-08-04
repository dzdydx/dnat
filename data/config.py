# -*- coding: utf-8 -*-
# @Time    : 2022-5-11 21:45:10
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn

datasets = ['audioset', 'fsd50k', 'esc50']
num_sec = { 'esc50': 5, 'audioset': 10 }
sample_rate = { 'esc50': 32000, 'audioset': 32000 }
target_length = { 'esc50': 498, 'audioset': 998 }
norm_mean = { 'esc50': -5.025409, 'audioset': -5 }
norm_std = { 'esc50': 5.4591165, 'audioset': 4.5 }
freq_mask = { 'esc50': 24, 'audioset': 48 }
time_mask = { 'esc50': 96, 'audioset': 192 }
mixup = { 'esc50': 0, 'audioset': 0.5 }

def get_dataset_conf(dataset_name, **kwargs):
    if dataset_name not in datasets:
        raise NotImplementedError(f"{dataset_name} not implemented.")
    
    mixup_rate = mixup.get(dataset_name)
    if kwargs["mixup_strategy"] == "no_mixup":
        mixup_rate = 0

    train_conf = {
        'num_sec': num_sec.get(dataset_name),
        'sample_rate': sample_rate.get(dataset_name, 32000),
        'num_mel_bins': kwargs.get("num_mel_bins", 128),
        'target_length': target_length.get(dataset_name),

        'skip_norm': False,
        'norm_mean': norm_mean.get(dataset_name),
        'norm_std': norm_std.get(dataset_name),

        'freq_mask': freq_mask.get(dataset_name),
        'time_mask': time_mask.get(dataset_name),
        'mixup': mixup_rate,
        'mixup_strategy': kwargs["mixup_strategy"]
    }

    val_conf = {
        'num_sec': num_sec.get(dataset_name),
        'sample_rate': sample_rate.get(dataset_name, 32000),
        'num_mel_bins': kwargs.get("num_mel_bins", 128),
        'target_length': target_length.get(dataset_name),

        'skip_norm': False,
        'norm_mean': norm_mean.get(dataset_name),
        'norm_std': norm_std.get(dataset_name),

        'freq_mask': 0,
        'time_mask': 0,
        'mixup': 0,
        'mixup_strategy': None
    }

    return train_conf, val_conf