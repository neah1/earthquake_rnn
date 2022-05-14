import asyncio
import logging
import os
import sys

import numpy as np
import pandas as pd
from obspy import read
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader


def mass_data_downloader(start, stop, event_id, Station, Network='NZ', Channel='HHZ', Location=10):
    """
    This function uses the FDSN mass data downloader to automatically download
    data from the XH network deployed on the RIS from Nov 2014 - Nov 2016.
    More information on the Obspy mass downloader available at:
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html
    Inputs:
    start: "YYYYMMDD"
    stop:  "YYYYMMDD"
    Network: 2-character FDSN network code
    Station: 2-character station code
    Channel: 3-character channel code
    Location: 10.
    """
    domain = RectangularDomain(
        minlatitude=-47.749,
        maxlatitude=-33.779,
        minlongitude=166.104,
        maxlongitude=178.990
    )
    restrictions = Restrictions(
        starttime=start,
        endtime=stop,
        chunklength_in_sec=None,
        network=Network,
        station=Station,
        location=Location,
        channel=Channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0
    )
    ev_str = str(event_id).replace(":", "_")
    try:
        mdl.download(domain, restrictions,
                     mseed_storage=f"./datasets/{folder}/waveforms/{ev_str}",
                     stationxml_storage=f"./datasets/{folder}/stations")
    except Exception as e:
        print(f'Event: {ev_str}. Error: {e}')
        pass


async def final_download_threaded(events, T, H):
    tasks = []
    for i, event in events.iterrows():
        event_id = event.event_id
        event_time = event['time']
        start = event_time - T - H
        end = event_time - H
        stations = ",".join([station.station_code for j, station in stations_df.iterrows()])
        tasks.append(asyncio.to_thread(mass_data_downloader, start, end, event_id, stations))
    await asyncio.gather(*tasks)


async def final_download():
    print(f'Downloading {folder} waves')
    counter = threads_at_once
    for event_sublist in [events_df[x:x + threads_at_once] for x in range(0, len(events_df), threads_at_once)]:
        print(f'Current batch: {counter}/{len(events_df)}')
        counter += threads_at_once
        await final_download_threaded(event_sublist, T_event, H_event)


def process_waves():
    print(f'Processing {folder} waves')
    path = f'./datasets/{folder}/waveforms/smi_nz.org.geonet/'
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
                print(f'Event: {event}, Station: {station_name}. Error: {e}')
                continue
            if min(station_data) == -14822981 or max(station_data) == -14822981:
                print('Corrupted data')
                continue
            else:
                station_data_arr[station_name] = station_data
        final_data[event] = station_data_arr
    final_data = pd.DataFrame(final_data).transpose()
    final_data.to_pickle(f'./datasets/{folder}/waves_temp.pkl')


mdl = MassDownloader(providers=['GEONET'])
logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
logger.setLevel(logging.WARNING)
threads_at_once = 100

# Parameters
folder = "active"
if folder == "active":
    events_df = pd.read_pickle('datasets/events_processed.pkl')
    H_event = 0
else:
    events_df = pd.read_pickle('datasets/events_normal.pkl')
    H_event = 2000
stations_df = pd.read_pickle('./datasets/stations_processed.pkl')

# TODO Active 500
# TODO Normal 500
# TODO Download larger sample
T_event = 30
events_df = events_df[500:2000]

asyncio.run(final_download())
process_waves()
