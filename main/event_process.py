from math import radians

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# TODO Unneeded
def add_closest(events):
    all_stations = []
    for j, event in events.iterrows():
        if j % 1000 == 0:
            print(f'{j}/{events.shape[0]}')
        all_distances = {}
        for i, station in stations_df.iterrows():
            e_coord = [radians(_) for _ in [event['latitude'], event['longitude']]]
            s_coord = [radians(_) for _ in [station['latitude'], station['longitude']]]
            distance = haversine_distances([e_coord, s_coord])[0][1]
            all_distances[station.station_code] = distance * 6371
        closest_stations = [k for (k, v) in all_distances.items() if v < 50]
        all_stations.append(closest_stations)
    events['closest_stations'] = all_stations
    events = events[events['closest_stations'].str.len() > 0]
    events.to_pickle('datasets/events_processed_new.pkl')


# TODO filter stations to 58
# TODO re-filter normal events from updated events
def filter_normal(events):
    new_events = pd.DataFrame({}, columns=events.columns)
    prev_time = None
    for i, event in events.iterrows():
        if i % 1000 == 0:
            print(f'{i}/{events.shape[0]}')
        time = event['time']
        if not prev_time:
            prev_time = time
            continue
        if (time - prev_time) > 10000:
            new_events.loc[len(new_events.index)] = event
        prev_time = time
    new_events.to_pickle('datasets/events_normal.pkl')

def sort_by_time(path='./datasets/events_normal.pkl'):
    events = pd.read_pickle(path)
    events = events.sort_values(by=['time'])
    events.to_pickle(path)


def plot_stations(stations, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    stations.plot(x="longitude", y="latitude", kind="scatter", ax=ax)
    ax.grid(visible=True, alpha=0.5)
    plt.savefig(f'./runs/{name}.png', transparent=True)
    # writer.add_figure(name, plt.gcf(), 0)
    del fig, ax


def plot_events(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    events.plot(x="longitude", y="latitude", kind="scatter", s=0.1, ax=ax)
    plt.savefig(f'./runs/{name}.png', transparent=True)
    # writer.add_figure(name, plt.gcf(), 0)
    del fig, ax


def plot_magnitude(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    w = 0.2
    n = np.ceil((events['magnitude'].max() - events['magnitude'].min()) / w)
    plt.hist(events['magnitude'], histtype='bar', ec='black', bins=int(n))
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Number of Earthquakes', fontsize=20)
    plt.xlabel('Magnitude (M)', fontsize=20)
    plt.xlim([0, 6])
    plt.savefig(f'./runs/{name}.png', transparent=True)
    # writer.add_figure(name, plt.gcf(), 0)
    del fig, ax


def plot_depth(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hist(events['depth'], histtype='bar', ec='black', bins=25)
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Number of Earthquakes', fontsize=20)
    plt.xlabel('Depth (km)', fontsize=20)
    plt.xlim([0, 500])
    plt.savefig(f'./runs/{name}.png', transparent=True)
    # writer.add_figure(name, plt.gcf(), 0)
    del fig, ax


events_df = pd.read_pickle('datasets/events_processed.pkl')
events_full = pd.read_pickle('datasets/events_processed_full.pkl')
stations_df = pd.read_pickle('./datasets/stations_processed.pkl')

# TODO add pngs to tensorboard
# tensorboard --logdir=main/runs
# writer = SummaryWriter("./runs/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
# writer.close()

plot_stations(stations_df, 'New Zealand stations')
plot_events(events_full, 'New Zealand earthquakes')
plot_events(events_df, 'Filtered earthquakes')
plot_magnitude(events_df, 'Earthquakes magnitudes')
plot_depth(events_df, 'Earthquakes depth')
