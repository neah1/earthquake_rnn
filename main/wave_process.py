import os
import pickle
import sys
from math import ceil, floor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from obspy import read
from sklearn.preprocessing import Normalizer


def sanitize(df):
    print(f'Before: {df.shape}')
    df = df.dropna()
    for i, row in df.iterrows():
        if (row.str.len() < 3001).any():
            df = df.drop(i)
    print(f'After: {df.shape}')
    return df


def set_labels():
    events = pd.read_pickle('./datasets/events_processed.pkl')
    high = events[events['magnitude'] > 2.5]['event_id'].apply(lambda x: x.split('/')[1])
    normal['label'] = 0
    active['label'] = active.index
    active['label'] = active['label'].apply(lambda x: 0 if x in list(high) else 1)


def combine_data(low, high, flat):
    set_labels()
    active_low = active[active['label'] == 1]
    active_high = active[active['label'] == 0]
    low_size = len(active_low)
    high_size = len(active_high)
    flat_size = len(normal)

    idx = min([low_size / low, high_size / high, flat_size / flat])
    dataset = pd.concat([active_low[:floor(idx * low)], active_high[:floor(idx * high)], normal[:floor(idx * flat)]])
    dataset.to_pickle('./datasets/sets/dataset.pkl')


active = sanitize(pd.read_pickle('./datasets/active/waves_full.pkl'))
normal = sanitize(pd.read_pickle('./datasets/normal/waves_full.pkl'))
combine_data(low=0.5, high=0.25, flat=0.25)

# TODO split (train, test, valid), shuffle (time-series), k-fold,


# TODO Normalize, scale, down-sample, sort based on time.
# norm = Normalizer()
# reduction_factor = 2  # TODO Try different HZ
# new_signal = ceil(3001 / reduction_factor)
# time_arr = np.linspace(0.0, 30.0, new_signal)

# norm.fit([item for items in all_data for item in items])
# all_data = [norm.transform(i) for i in all_data]
# pickle.dump(all_data, open(f'./datasets/{folder}/{round(100 / reduction_factor)}hz.pkl', "wb"))

# def preprocess_data(data):
#     res = []
#     for i, new_data in enumerate(data):
#         new_data = new_data[::reduction_factor]
#         new_data = new_data[:new_signal]
#         res.append(new_data)
#     return res


# TODO Plot samples
# samples = norm.fit_transform(preprocess_data(data))
# plot_waves(samples, f'Sample waves')

# def plot_waves(data, name):
#     fig = plt.figure(figsize=(15, 6))
#     ax = fig.add_subplot(1, 1, 1)
#     cmap = plt.get_cmap('Accent')
#     colors = [cmap(i) for i in np.linspace(0, 1, len(data))]
#     for i, s in enumerate(data):
#         ax.plot(time_arr, s, color=colors[i])
#     ax.tick_params(axis='both', labelsize=15)
#     ax.tick_params(axis='both', labelsize=15)
#     plt.ylabel('Velocity HZ', fontsize=20)
#     plt.xlabel('Timestep', fontsize=20)
#     plt.xlim([0, 30])
#     plt.savefig(f'./runs/{name}.png', transparent=True)
#     del fig, ax
