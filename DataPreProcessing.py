from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
from operator import mul
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from haversine import haversine
from datetime import datetime
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading the Input Data and sorting it in ascending order of 't_user_id' and 'collected_time'. This way of sorting
# the data is equivalent to grouping the data on the bases of 't_user_id' and 'collected_time'.
df = pd.read_csv('geolife_raw.csv')
total = len(df)
df = df.sort_values(['t_user_id', 'collected_time'], ascending=True).reset_index(drop=True)

# Preprocessing
x = df['collected_time'].str.split(' ', expand = True)
df['date_Start'] = x[0]
y = x[1].str.split('-', expand = True)
df['time_Start'] = y[0]
df['latitude_Start'] = df['latitude']
df['longitude_Start'] = df['longitude']

df['latitude_End'] = df['latitude'].drop([0]).reset_index(drop=True)
df['longitude_End'] = df['longitude'].drop([0]).reset_index(drop=True)

df['date_End'] = df['date_Start'].drop([0]).reset_index(drop=True)
df['time_End'] = df['time_Start'].drop([0]).reset_index(drop=True)

df['UserChk'] = df['t_user_id'].drop([0]).reset_index(drop=True)
df['ModeChk'] = df['transportation_mode'].drop([0]).reset_index(drop=True)
df = df.drop('collected_time', axis =1)
df = df.drop('latitude', axis =1)
df = df.drop('longitude', axis =1)
df = df.drop([total-1], axis =0)

# The above preprocessing has created a DataFrame with the columns arranged in a much more better computational way.
# Columns of processable data now are :- 
# 't_user_id', 'transportation_mode', 'date_Start', 'time_Start', 'latitude_Start', 'longitude_Start',
# 'latitude_End', 'longitude_End','date_End', 'time_End', 'UserChk', 'ModeChk' 
# We finally convert this DataFrame to list of lists because of the better time complexity of lists than pandas DataFrame
dataList = df.values.tolist()
