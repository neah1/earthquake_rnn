import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt


def plot_stations(events, stations, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    events.plot(x="longitude", y="latitude", kind="scatter", s=0.1, ax=ax)
    stations.plot(x="longitude", y="latitude", kind="scatter", ax=ax, color='yellow')
    plt.savefig(f'./plots/{name}.png', transparent=True)
    del fig, ax


def plot_events(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    events.plot(x="longitude", y="latitude", kind="scatter", s=0.1, ax=ax)
    plt.savefig(f'./plots/{name}.png', transparent=True)
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
    plt.savefig(f'./plots/{name}.png', transparent=True)
    del fig, ax


def plot_depth(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hist(events['depth'], histtype='bar', ec='black', bins=25)
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Number of Earthquakes', fontsize=20)
    plt.xlabel('Depth (km)', fontsize=20)
    plt.xlim([0, 500])
    plt.savefig(f'./plots/{name}.png', transparent=True)
    del fig, ax


def plot_waves(data, name, signal=3000):
    time_arr = np.linspace(0.0, 30.0, signal)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap('Accent')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data))]
    for i, s in enumerate(data):
        ax.plot(time_arr, s, color=colors[i])
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Velocity HZ', fontsize=20)
    plt.xlabel('Timestep', fontsize=20)
    plt.xlim([0, 30])
    plt.savefig(f'./plots/{name}.png', transparent=True)
    del fig, ax

events_df = pd.read_pickle('./datasets/sets/events_processed.pkl')
stations_df = pd.read_pickle('./datasets/sets/stations_processed.pkl')
waves_df = pd.read_pickle('./datasets/sets/dataset.pkl')

# plot_stations(events_df, stations_df, 'NZ stations')
# plot_magnitude(events_df, 'Event magnitudes')
# plot_events(events_df, 'NZ earthquakes')
# plot_depth(events_df, 'event depths')
plot_waves(waves_df[waves_df['label'] == 0].iloc[-2][0:3], 'waves')
