import sys
from math import floor
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler


def sanitize(df, frames):
    pd.options.mode.chained_assignment = None
    frames = frames * 100
    print(f'Before: {df.shape}')
    df = df.dropna()
    for i, row in df.iterrows():
        if (row.str.len() < frames).any():
            df = df.drop(i)
            continue
        df.loc[i] = row.apply(lambda x: x[-frames:])
    print(f'After: {df.shape}')
    return df


def combine_data(low, high, flat):
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
    return pd.concat([active_low[:floor(idx * low)], active_high[:floor(idx * high)], normal[:floor(idx * flat)]])


def normalize_scale(df):
    temp = df['label'].copy()
    df = df.drop(columns=['label'])
    norm = Normalizer()
    norm.fit(df.values.flatten().tolist())
    df = df.apply(lambda x: x.apply(lambda y: norm.transform(y.reshape(1, -1))[0]))
    scale = StandardScaler()
    scale.fit(df.values.flatten().tolist())
    df = df.apply(lambda x: x.apply(lambda y: scale.transform(y.reshape(1, -1))[0]))
    df['label'] = temp
    return df


active = sanitize(pd.read_pickle('./datasets/active/waves_full.pkl'), 30)
normal = sanitize(pd.read_pickle('./datasets/normal/waves_full.pkl'), 30)
dataset = combine_data(low=0.5, high=0.0, flat=0.5)
dataset = normalize_scale(dataset)
dataset.to_pickle('./datasets/sets/dataset.pkl')
