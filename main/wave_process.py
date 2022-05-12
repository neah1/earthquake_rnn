import os
import pickle
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from obspy import read
from sklearn.preprocessing import Normalizer


def plot_wave_processed(data, name):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap('Accent')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data))]
    for i, s in enumerate(data):
        ax.plot(time_arr, s, color=colors[i])
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('HHZ Velocity', fontsize=20)
    plt.xlabel('Timestep', fontsize=20)
    plt.xlim([0, 30])
    plt.savefig(f'./runs/{name}.png', transparent=True)
    # writer.add_figure(name, plt.gcf(), 0)
    del fig, ax


def preprocess_data(data):
    res = []
    for i, new_data in enumerate(data):
        new_data = new_data[::reduction_factor]
        new_data = new_data[:new_signal]
        res.append(new_data)
    return res


def get_samples(space):
    res = []
    for i in space:
        inter = path + os.listdir(path)[i] + '/'
        file = inter + os.listdir(inter)[0]
        res.append(read(file)[0].data)
    return res


def process_all():
    final_data = []
    list_of_directories = os.listdir(path)
    for j, directory in enumerate(list_of_directories):
        if j % 100 == 0:
            print(f'{j}/{len(list_of_directories)}')
        # stations for each event
        cur_dir = path + directory
        station_files = os.listdir(cur_dir)
        station_files = [os.path.join(cur_dir, station) for station in station_files]
        # station data per event
        station_data_arr = []
        for i in range(0, len(station_files)):
            try:
                station_data = read(station_files[i])[0].data
            except:
                continue
            # if corrupted data, continue to next station
            if min(station_data) == -14822981 or max(station_data) == -14822981 or len(station_data) < og_signal:
                continue
            else:
                station_data_arr.append(station_data)
        if station_data_arr:
            res = preprocess_data(station_data_arr)
            final_data.append(res)
    return final_data


# Parameters
folder = 'normal'
path = f'./datasets/{folder}/waveforms/smi_nz.org.geonet/'
norm = Normalizer()
og_signal = 3001
# TODO tweak down-sampling
reduction_factor = 10
new_signal = ceil(og_signal / reduction_factor)
time_arr = np.linspace(0.0, 30.0, new_signal)

# Process data
all_data = process_all()
norm.fit([item for items in all_data for item in items])
all_data = [norm.transform(i) for i in all_data]
save_loc = f'./datasets/{folder}/'
pickle.dump(all_data, open(save_loc + f'{folder}_10hz.pkl', "wb"))

# Plot samples
samples = norm.fit_transform(preprocess_data(get_samples([20, 21, 22])))
plot_wave_processed(samples, f'Sample waves')
