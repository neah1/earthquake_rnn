import os
from math import floor

import numpy as np
import pandas as pd
from obspy import read
from sklearn.preprocessing import Normalizer, StandardScaler


def process_waves(folder):
    print(f'Processing {folder} waves')
    path = f'./datasets/{folder}/waveforms/'
    events = os.listdir(path)
    final_data = {}
    for j, event in enumerate(events):
        if j % 100 == 0:
            print(f'Current batch: {j}/{len(events)}')
        cur_dir = path + event
        station_files = os.listdir(cur_dir)
        station_data_arr = {}
        for station in station_files:
            station_name = station.split('.')[1]
            try:
                file_name = cur_dir + '/' + station
                station_data = np.array(read(file_name)[0].data)
            except Exception as e:
                print(e)
                continue
            if min(station_data) == -14822981 or max(station_data) == -14822981:
                continue
            else:
                station_data_arr[station_name] = station_data
        final_data[event] = station_data_arr
    final_data = pd.DataFrame(final_data).transpose()
    final_data.to_pickle(f'./datasets/{folder}/waves_temp.pkl')


def join_waves(folder):
    print(f'Joining {folder} waves')
    temp = f'./datasets/{folder}/waves_temp.pkl'
    full = f'./datasets/{folder}/waves_full.pkl'
    if os.path.exists(full):
        df_temp = pd.read_pickle(temp)
        df_full = pd.read_pickle(full)
        df_full = pd.concat([df_full, df_temp])
        df_full.to_pickle(full)
        os.remove(temp)
    else:
        os.rename(temp, full)


def sanitize(folder, frames):
    pd.options.mode.chained_assignment = None
    df = pd.read_pickle(f'./datasets/{folder}/waves_full.pkl')
    print(f'Before {folder}: {df.shape}')
    frames = frames * 100
    df = df.dropna()
    for i, row in df.iterrows():
        if (row.str.len() < frames).any():
            df = df.drop(i)
            continue
        df.loc[i] = row.apply(lambda x: x[-frames:])
    print(f'After {folder}: {df.shape}')
    df.to_pickle(f'./datasets/{folder}/waves_full.pkl')


def combine_data(low, high, flat):
    print('Combining data')
    active = pd.read_pickle('./datasets/active/waves_full.pkl')
    normal = pd.read_pickle('./datasets/normal/waves_full.pkl')
    events = pd.read_pickle('./datasets/sets/events_processed.pkl')

    high_events = events[events['magnitude'] > 2.5]['event_id'].apply(lambda x: x.split('/')[1])
    normal['label'] = 0
    active['label'] = active.index
    active['label'] = active['label'].apply(lambda x: 0 if x in list(high_events) else 1)
    active_low = active[active['label'] == 1]
    active_high = active[active['label'] == 0]

    inf = float('inf')
    low_size = inf if low == 0.0 else len(active_low) / low
    high_size = inf if high == 0.0 else len(active_high) / high
    flat_size = inf if flat == 0.0 else len(normal) / flat
    idx = min([low_size, high_size, flat_size])
    print(f'IDX: {idx}, Low events: {len(active_low)}, High events: {len(active_high)}, Normal events: {len(normal)}')
    df = pd.concat([active_low[:floor(idx * low)], active_high[:floor(idx * high)], normal[:floor(idx * flat)]])
    df.to_pickle('./datasets/sets/dataset.pkl')


def normalize_scale(scale, normalize):
    print('Normalizing dataset')
    df = pd.read_pickle('./datasets/sets/dataset.pkl')
    temp = df['label'].copy()
    df = df.drop(columns=['label'])
    if scale:
        norm = Normalizer()
        norm.fit(df.values.flatten().tolist())
        df = df.apply(lambda x: x.apply(lambda y: norm.transform(y.reshape(1, -1))[0]))
    if normalize:
        scaler = StandardScaler()
        scaler.fit(df.values.flatten().tolist())
        df = df.apply(lambda x: x.apply(lambda y: scaler.transform(y.reshape(1, -1))[0]))
    df['label'] = temp
    df.to_pickle('./datasets/sets/dataset.pkl')

# process_waves('active')
# join_waves('active')
# sanitize('active', 60)

# process_waves('normal')
# join_waves('normal')
# sanitize('normal', 60)

combine_data(low=0.5, high=0.0, flat=0.5)

# normalize_scale(scale=True, normalize=True)
