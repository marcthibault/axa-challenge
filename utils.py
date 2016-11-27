import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from datetime import datetime, date


def create_matrix(df, weekday):
    result=np.array([])
    dates = df.index
    first_day = dates[0].weekday()
    decal = (7-first_day+weekday)%7

    first_day = dates[decal*48]
    last_day = dates[-7*48].date()

    empty_weeks = [(datetime(year=2012,month=12,day=28),datetime(year=2013,month=1,day=4)),
                   (datetime(year=2013,month=2,day=2),datetime(year=2013,month=2,day=9)),
                   (datetime(year=2013,month=3,day=6),datetime(year=2013,month=3,day=13)),
                   (datetime(year=2013,month=4,day=10),datetime(year=2013,month=4,day=17)),
                   (datetime(year=2013,month=5,day=13),datetime(year=2013,month=5,day=20)),
                   (datetime(year=2013,month=6,day=12),datetime(year=2013,month=6,day=19)),
                   (datetime(year=2013,month=7,day=16),datetime(year=2013,month=7,day=23)),
                   (datetime(year=2013,month=8,day=15),datetime(year=2013,month=8,day=22)),
                   (datetime(year=2013,month=9,day=14),datetime(year=2013,month=9,day=21)),
                   (datetime(year=2013,month=10,day=18),datetime(year=2013,month=10,day=25)),
                   (datetime(year=2013,month=11,day=20),datetime(year=2013,month=11,day=27)),
                   (datetime(year=2013,month=12,day=22),datetime(year=2013,month=12,day=29))]

    while True:
        if (first_day+pd.Timedelta('7 day')).date()>last_day:
            break
        arr = np.array(df.loc[first_day:first_day+pd.Timedelta('7 day')][:-1].fillna(method="bfill").fillna(method="ffill").transpose().values)
        if np.sum(np.isnan(arr)) >0:
            arr = np.zeros(shape=arr.shape)
        for (d1,d2) in empty_weeks:
            if (first_day>=d1 and first_day<d2) or (first_day+pd.Timedelta('7 day')>=d1 and first_day+pd.Timedelta('7 day')<d2):
                arr = np.nan*np.zeros(shape=arr.shape)

        if len(result) == 0:
            result = arr
        else:
            result = np.append(result, arr, axis=0)

        first_day+=pd.Timedelta('7 day')
    return result
