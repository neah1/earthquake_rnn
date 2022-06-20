import pandas as pd
from tbparse import SummaryReader


temp = pd.read_pickle('datasets/waves/short_scale.pkl')
print(temp.iloc[0])
