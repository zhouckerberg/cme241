import pandas as pd
import os
import sys
import dask.dataframe as dd
import numpy as np

def load_data(path, start_date, end_date, prefix, suffix, date_format, sort_by=['t','ticker']):
    data = []
    for date in pd.date_range(start_date, end_date):
        filename = prefix + date.strftime(date_format) + suffix
        fullpath = os.path.join(path, filename)
        if os.path.isfile(fullpath):
            data.append(dd.read_csv(fullpath))
    if(len(data)==0):
        print('No data within the selected dates')
        return pd.DataFrame()
    data = dd.concat(data)
    data = data.compute()
    data.sort_values(by=sort_by, inplace=True)
    data.reset_index(0, drop=True, inplace=True)
    return data

def index(data, timeCol='t', indexCol='datetime', unit='s'):
    data.set_index(pd.to_datetime(data[timeCol], unit=unit).rename(indexCol), inplace=True)
    data.sort_values(by=indexCol, inplace=True)

def get_keys():
    return {'Y':'year','M':'month', 'D':'day', 'h':'hour', 'm':'minute', 's':'second', 'ms':'microsecond', 'dow':'dayofweek'}

def get(tx, attr):
    if attr in get_keys().keys():
        attr = get_keys()[attr]
    return getattr(tx.index,attr)

def compute_return(data, priceName, colName, lag):
    data[colName] =  data.groupby(['ticker']).apply(lambda x: pd.DataFrame({colName:10000*np.sign(lag)*x[priceName].pct_change(periods=lag)}, index = x.index))



