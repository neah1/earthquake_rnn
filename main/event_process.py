import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
        if (time - prev_time) > 5000:
            new_events.loc[len(new_events.index)] = event
        prev_time = time
    new_events.to_pickle('datasets/events_temp.pkl')


def filter_station(stations):
    selected_stations = ['BFZ', 'BKZ', 'DCZ', 'DSZ', 'EAZ', 'HIZ', 'JCZ', 'KHZ', 'KNZ', 'KUZ', 'LBZ', 'LTZ', 'MLZ',
                         'MQZ', 'MRZ', 'MSZ', 'MWZ', 'MXZ', 'NNZ', 'ODZ', 'OPRZ', 'OUZ', 'PUZ', 'PXZ', 'QRZ', 'RPZ',
                         'SYZ', 'THZ', 'TOZ', 'TSZ', 'TUZ', 'URZ', 'VRZ', 'WCZ', 'WHZ', 'WIZ', 'WKZ', 'WVZ']
    new_stations = stations.loc[stations.station_code.isin(selected_stations)]
    new_stations.to_pickle('datasets/stations_temp.pkl')


def filter_events(events):
    events = events[events['magnitude'] > 0.5]
    events.to_pickle('./datasets/events_temp.pkl')


def sort_by_time(events):
    events = events.sort_values(by=['time'])
    events.to_pickle('./datasets/events_temp.pkl')


def plot_stations(events, stations, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    events.plot(x="longitude", y="latitude", kind="scatter", s=0.1, ax=ax)
    stations.plot(x="longitude", y="latitude", kind="scatter", ax=ax, color='yellow')
    plt.savefig(f'./runs/{name}.png', transparent=True)
    del fig, ax


def plot_events(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    events.plot(x="longitude", y="latitude", kind="scatter", s=0.1, ax=ax)
    plt.savefig(f'./runs/{name}.png', transparent=True)
    del fig, ax


def plot_magnitude(events, name):
    fig, ax = plt.subplots(figsize=(12, 10))
    w = 0.2
    n = np.ceil((events['magnitude'].max() - events['magnitude'].min()) / w)
    plt.hist(events['magnitude'], histtype='bar', ec='black', bins=int(n))
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Number of Earthquakes', fontsize=50)
    plt.xlabel('Magnitude (M)', fontsize=50)
    plt.xlim([0, 5])
    plt.savefig(f'./runs/{name}.png', transparent=True)
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
    del fig, ax


events_df = pd.read_pickle('datasets/events_processed.pkl')
stations_df = pd.read_pickle('./datasets/stations_processed.pkl')

filter_station(stations_df)

# plot_stations(events_df, stations_df, 'New Zealand stations')
# plot_events(events_df, 'New Zealand earthquakes')
# plot_magnitude(events_df, 'Earthquakes magnitudes')
# plot_depth(events_df, 'Earthquakes depth')
