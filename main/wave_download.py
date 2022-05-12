import asyncio
import logging

import pandas as pd
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
    mdl.download(
        domain,
        restrictions,
        mseed_storage=f"./datasets/{folder}/waveforms/{ev_str}",
        stationxml_storage=f"./datasets/{folder}/stations",
    )


# TODO Redl active and normal waves
async def final_download_threaded(events, T, H):
    tasks = []
    for i, event in events.iterrows():
        event_id = event.event_id
        event_time = event['time']
        start = event_time - T - H
        end = event_time - H
        # TODO DL all stations
        station = ",".join(event['closest_stations'])
        tasks.append(asyncio.to_thread(mass_data_downloader, start, end, event_id, station))
    await asyncio.gather(*tasks)


async def final_download():
    for event_sublist in [events_df[x:x + threads_at_once] for x in range(0, len(events_df), threads_at_once)]:
        print("Next batch")
        await final_download_threaded(event_sublist, T_event, H_event)


events_df = None
stations_df = pd.read_pickle('./datasets/stations_processed.pkl')
mdl = MassDownloader(providers=['GEONET'])
logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
logger.setLevel(logging.WARNING)

# Parameters
folder = "normal"
if folder == "active":
    events_df = pd.read_pickle('datasets/events_processed.pkl')
    H_event = 0
elif folder == "normal":
    events_df = pd.read_pickle('datasets/events_normal.pkl')
    H_event = 5000
T_event = 30
threads_at_once = 100
events_df = events_df[0:1000]

asyncio.run(final_download())
