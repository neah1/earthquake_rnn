from obspy.clients.fdsn import Client as FDSN_Client
from datetime import timedelta, datetime
import pandas as pd


def download_events(starttime, endtime):
    print('starting download from ', starttime, ' to ', endtime)
    cat = client.get_events(starttime=starttime, endtime=endtime, minlatitude=-47.749, maxlatitude=-33.779,
                            minlongitude=166.104, maxlongitude=178.990)
    print("done")
    return cat


def get_all_events(starttime, endtime, step):
    cat = None
    cur = starttime
    while cur < endtime:
        next_time = cur + step
        if next_time > endtime:
            next_time = endtime
        cur_cat = download_events(cur, next_time)
        if cat is None:
            cat = cur_cat
        else:
            cat.extend(cur_cat)
        cur = next_time
    print("done with all downloads")
    return cat


def save_cat(cat):
    event_ids = []
    event_times = []
    latitudes = []
    longitudes = []
    magnitudes = []
    depths = []

    for i in range(len(cat)):
        earthquakeEvent = cat[i]

        event_id = earthquakeEvent.resource_id.id
        event_time = earthquakeEvent.preferred_origin().time
        latitude = earthquakeEvent.preferred_origin().latitude
        longitude = earthquakeEvent.preferred_origin().longitude
        magnitude = round(earthquakeEvent.preferred_magnitude().mag, 1)
        depth = round(earthquakeEvent.preferred_origin().depth / 1000)

        event_ids.append(event_id)
        event_times.append(event_time)
        latitudes.append(latitude)
        longitudes.append(longitude)
        magnitudes.append(magnitude)
        depths.append(depth)

    data_map = {'event_id': event_ids, 'time': event_times, 'latitude': latitudes, 'longitude': longitudes,
                'magnitude': magnitudes, 'depth': depths}
    df = pd.DataFrame(data=data_map)
    df.to_pickle('events/events.pkl')


client = FDSN_Client("GEONET")
cat = get_all_events(datetime(1999, 12, 31), datetime(2003, 12, 31), timedelta(days=50))
save_cat(cat)