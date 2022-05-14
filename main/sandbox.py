import pandas as pd

temp = pd.read_pickle('./datasets/normal/waves_temp.pkl')
events = pd.read_pickle('./datasets/events_processed.pkl')

print(temp.shape)

