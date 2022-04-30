from dl_data import get_all_events, save_cat
from datetime import timedelta, datetime

cat = get_all_events(datetime(1999, 12, 31), datetime(2003, 12, 31), timedelta(days=50))
save_cat(cat)
