import mpl_toolkits
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import logging
from obspy import read

from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from scipy import signal
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import asyncio

events_df = pd.read_pickle('data/events_processed.pkl')
stations_df = pd.read_pickle('data/stations_processed.pkl')

events_full = events_df[(events_df.time > '2016-01-01') & (events_df.time < '2017-01-01')]
events_full.shape
events = events_full[10:20]
mdl = MassDownloader(providers=['GEONET'])


def mass_data_downloader(start, stop, event_id, Station,
                         Network='NZ',
                         Channel='HHZ',
                         Location=10
                         ):
    '''
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
    '''
    # print("=" * 65)
    # print("Initiating mass download request.")

    domain = RectangularDomain(
        minlatitude=-47.749,
        maxlatitude=-33.779,
        minlongitude=166.104,
        maxlongitude=178.990
    )

    restrictions = Restrictions(
        starttime=start,
        endtime=stop,
        # 24 hr
        chunklength_in_sec=86400,
        network=Network,
        station=Station,
        location=Location,
        channel=Channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0
    )

    # mdl = MassDownloader(providers=['GEONET'])
    ev_str = str(event_id).replace(":", "_")
    mdl.download(
        domain,
        restrictions,
        mseed_storage=f"datasets/normal/waveforms/{ev_str}",
        stationxml_storage="datasets/normal/stations",
    )


logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
logger.setLevel(logging.WARNING)


async def final_download():
    for i, event in events.iterrows():
        event_id = event.event_id
        event_time = event['time']
        start = event_time - 30
        end = event_time

        print("=" * 65)
        print("Initiating mass download request.")
        print(event_id)

        tasks = [asyncio.to_thread(mass_data_downloader, start, end, event_id, station.station_code) for j, station in
                 stations_df.iterrows()]
        await asyncio.gather(*tasks)


if __name__ == '__main__':
    await final_download()