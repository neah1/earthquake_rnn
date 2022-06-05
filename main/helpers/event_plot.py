import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def plot_events(events, stations, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = countries[countries["name"] == "New Zealand"].plot(color="lightgrey", ax=ax)
    events.plot(x="longitude", y="latitude", kind="scatter", s=0.1, color='yellow', alpha=0.15, ax=ax)
    stations.plot(x="longitude", y="latitude", kind="scatter", ax=ax)
    plt.xlabel('Longitude', fontsize=20)
    plt.ylabel('Latitude', fontsize=20)
    plt.xlim([166, 179])
    plt.ylim([-48, -34])
    plt.grid(color="grey")
    plt.savefig(f'../datasets/plots/{name}.png', transparent=True)
    del fig, ax


def plot_magnitude(events, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    n = np.ceil((events['magnitude'].max() - events['magnitude'].min()) / 0.2)
    plt.hist(events['magnitude'], histtype='bar', ec='black', bins=int(n))
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Number of Earthquakes', fontsize=20)
    plt.xlabel('Magnitude (M)', fontsize=20)
    plt.xlim([0, 5])
    plt.savefig(f'../datasets/plots/{name}.png', transparent=True)
    del fig, ax


def gather_event_plots():
    events_df = pd.read_pickle('../datasets/sets/events.pkl')
    stations_df = pd.read_pickle('../datasets/sets/stations.pkl')
    plot_events(events_df, stations_df, 'stations')
    plot_magnitude(events_df, 'magnitude')

    events_df = pd.read_pickle('../datasets/sets/events_full.pkl')
    stations_df = pd.read_pickle('../datasets/sets/stations_full.pkl')
    plot_events(events_df, stations_df, 'all_stations')
    plot_magnitude(events_df, 'all_magnitude')


def plot_waves(data, name, low, high, signal=60):
    time_arr = np.linspace(0.0, signal, signal * 100)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap('Accent')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data))]
    for i, s in enumerate(data):
        ax.plot(time_arr, s, color=colors[i])
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('HHZ Velocity', fontsize=20)
    plt.xlabel('Timestep', fontsize=20)
    plt.xlim([0, signal])
    plt.ylim([low, high])
    plt.savefig(f'../datasets/plots/{name}.png', transparent=True)
    del fig, ax


def gather_wave_plots(name, low, high):
    df = pd.read_pickle(f'../datasets/data/{name}.pkl')
    plot_waves(df.iloc[0][15:25], f'{name}_low', low, high)
    plot_waves(df.iloc[1][15:25], f'{name}_high', low, high)
    plot_waves(df.iloc[2][15:25], f'{name}_flat', low, high)


# gather_wave_plots('temp_norm', -0.15, 0.17)
# gather_wave_plots('temp_scale', -9, 2)
# gather_wave_plots('temp_both', -4, 6)
