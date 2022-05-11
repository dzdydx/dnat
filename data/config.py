datasets = ['audioset', 'fsd50k', 'esc50']
sample_rate = { 'esc50': 44100, }
time_frames = { 'esc50': 498, }
norm_mean = { 'esc50': -5.025409, }
norm_std = { 'esc50': 5.4591165, }
freq_mask = { 'esc50': 24, }
time_mask = { 'esc50': 96, }
mixup = { 'esc50': 0, }

def get_dataset_conf(dataset_name, **kwargs):
    if dataset_name not in datasets:
        raise NotImplementedError(f"{dataset_name} not implemented.")

    train_conf = {
        'sample_rate': sample_rate.get(dataset_name, 32000),
        'num_mel_bins': kwargs.get("num_mel_bins", 128),
        'time_frames': time_frames.get(dataset_name),

        'skip_norm': False,
        'norm_mean': norm_mean.get(dataset_name),
        'norm_std': norm_std.get(dataset_name),

        'freq_mask': freq_mask.get(dataset_name),
        'time_mask': time_mask.get(dataset_name),
        'mixup': mixup.get(dataset_name)
    }

    val_conf = {
        'sample_rate': sample_rate.get(dataset_name, 32000),
        'num_mel_bins': kwargs.get("num_mel_bins", 128),
        'time_frames': time_frames.get(dataset_name),

        'skip_norm': False,
        'norm_mean': norm_mean.get(dataset_name),
        'norm_std': norm_std.get(dataset_name),

        'freq_mask': 0,
        'time_mask': 0,
        'mixup': 0
    }

    return train_conf, val_conf