import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import random
from obspy import read


def check_nans_incomplete_stations(data):
    nan_arr = []
    wrong_stations = []
    wrong_samples = []
    for i in range(0, len(data)):
        if len(data[i]) != station_count:
            wrong_stations.append(i)
        for j in range(0, station_count):
            if len(data[i][j]) != signal_samples:
                wrong_samples.append(i)
            if np.isnan(np.sum(data[i][j])):
                nan_arr.append(i)
    return nan_arr, wrong_stations, wrong_samples


def verify_unique_shapes(data):
    stations_shape = [station_count]
    samples_shape = [signal_samples]
    for i in data:
        station_shape = len(i)
        if station_shape not in stations_shape:
            stations_shape.append(station_shape)
        for j in i:
            sample_shape = len(j)
            if sample_shape not in samples_shape:
                samples_shape.append(sample_shape)
    print(f'Stations: {stations_shape}, Samples: {samples_shape}, Length: {len(data)}')


def remove_incomplete(data):
    verify_unique_shapes(data)
    nan_arr, wrong_stations, wrong_samples = check_nans_incomplete_stations(data)
    remove_indices = [y for x in [nan_arr, wrong_stations, wrong_samples] for y in x]
    remove_indices = list(set(remove_indices))
    data = [i for j, i in enumerate(data) if j not in remove_indices]
    verify_unique_shapes(data)
    return data


def to_lstm_input(data, label):
    lstm_event_arr = []
    for i, r in enumerate(data):
        arr = np.transpose(data[i])
        arr = arr.reshape(signal_samples, station_count)
        lstm_event_arr.append({label: arr})
    return np.array(lstm_event_arr)


def shuffle(test_list1, test_list2):
    c1 = 0
    c2 = 0
    out = []
    n = len(test_list1) + len(test_list2)
    for i in range(n):
        diff = 10 * (c1 - c2) / n
        ledge = max(min(0.5 + diff, 1), 0)
        if random.random() > ledge and c1 < len(test_list1):
            out.append(test_list1[c1])
            c1 += 1
        elif c2 < len(test_list2):
            out.append(test_list2[c2])
            c2 += 1
    print(c1 / c2)
    out = np.concatenate((out, test_list1[c1:]))
    out = np.concatenate((out, test_list2[c2:]))
    return np.array(out)


def regular_split(data):
    full_n = len(data)
    train_n = round(full_n * train_ratio)
    valid_n = train_n + round(full_n * valid_ratio)
    return data[:train_n], data[train_n:valid_n], data[valid_n:]


def prepare_final(data, name):
    x = []
    y = []
    for i, r in enumerate(data):
        arr = data[i]
        key = list(arr.keys())[0]
        if key == '1':
            x.append(1)
        else:
            y.append(0)
        x.append(arr[key])
    pickle.dump(x, open(f"./datasets/sets/x_{name}.pkl", "wb"))
    pickle.dump(y, open(f"./datasets/sets/y_{name}.pkl", "wb"))


def k_fold(k):
    training_set = []
    validation_set = []
    test_set = []
    splits = np.array_split(lstm_input, k)
    for split in splits:
        full_n = len(split)
        train_n = round(full_n * train_ratio)
        valid_n = train_n + round(full_n * valid_ratio)
        training_set.append(split[:train_n])
        validation_set.append(split[train_n:valid_n])
        test_set.append(split[valid_n:])
    return np.array(training_set).flatten(), np.array(validation_set).flatten(), np.array(test_set).flatten()


active_data = pd.read_pickle(f'./datasets/active/active_10hz.pkl')
normal_data = pd.read_pickle(f'./datasets/normal/normal_10hz.pkl')
signal_samples = 301
station_count = 1
train_ratio = 0.7
valid_ratio = 0.1

active_data = remove_incomplete(active_data)
normal_data = remove_incomplete(normal_data)

true_data = to_lstm_input(active_data, '1')
false_data = to_lstm_input(normal_data, '0')
# TODO Check shuffling. CHECK EVERYTHING
lstm_input = np.concatenate((true_data, false_data))
np.random.shuffle(lstm_input)
train, valid, test = regular_split(lstm_input)
prepare_final(train, 'train')
prepare_final(valid, 'valid')
prepare_final(test, 'test')
