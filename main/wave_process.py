from math import floor
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler


def sanitize(df):
    pd.options.mode.chained_assignment = None
    print(f'Before: {df.shape}')
    df = df.dropna()
    for i, row in df.iterrows():
        if (row.str.len() < 3001).any():
            df = df.drop(i)
            continue
        df.loc[i] = row.apply(lambda x: x[:3001])
    print(f'After: {df.shape}')
    return df


def set_labels():
    events = pd.read_pickle('datasets/sets/events_processed.pkl')
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
    return pd.concat([active_low[:floor(idx * low)], active_high[:floor(idx * high)], normal[:floor(idx * flat)]])


# TODO Check for bugs
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


active = sanitize(pd.read_pickle('./datasets/active/waves_full.pkl'))
normal = sanitize(pd.read_pickle('./datasets/normal/waves_full.pkl'))

dataset = combine_data(low=0.5, high=0.25, flat=0.25)
dataset = normalize_scale(dataset)
dataset.to_pickle('./datasets/sets/dataset.pkl')
