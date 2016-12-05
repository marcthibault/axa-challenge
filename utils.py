import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from datetime import datetime, date

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

def approx_svd(mat, k):
    U,s,V = np.linalg.svd(mat, full_matrices=False)
    S = np.diag(s[0:k])
    return np.dot(U[:,0:k], np.dot(S, V[0:k,:]))

def create_training_xy_day(df_week):
    '''
    Create the matrix X and Y for the ML algorithms. Create here an Y corresponding to an entire day
    '''
    X = np.array([])
    Y = np.array([])

    for w in range(0,len(df_week)-1):
        if np.sum(np.isnan(df_week[w]))==0 and np.sum(np.isnan(df_week[w+1,-48:]))==0:
            if len(X)==0:
                X = np.array([df_week[w]])
                Y = np.array([df_week[w+1,-48:]])
            else:
                X = np.append(X, [df_week[w]], axis=0)
                Y = np.append(Y, [df_week[w+1,-48:]], axis=0)

    return X,Y

def create_training_xy_hour(df_week, hour, minute):
    '''
    Create the matrix X and Y for the ML algorithms. Create here an Y corresponding to a unique slot
    '''
    X = np.array([])
    Y = np.array([])

    for w in range(0,len(df_week)-1):
        if np.sum(np.isnan(df_week[w]))==0 and np.sum(np.isnan(df_week[w+1,-48:]))==0:
            if len(X)==0:
                X = np.array([df_week[w]])
                Y = np.array([df_week[w+1,-48+2*hour+int(minute/30)]])
            else:
                X = np.append(X, [df_week[w]], axis=0)
                Y = np.append(Y, df_week[w+1,-48+2*hour+int(minute/30)])

    return X,Y

def create_day_matrix(df, weekday, day_max = -1):
    '''
    Create the matrix where a row corresponds to the day of the week equaling weekday, and each column to a time slot
    '''

    result=np.array([])

    # we start at week 1 of 2011
    dates = df.index[96:]
    first_day = dates[0].weekday()
    decal = (7-first_day+weekday)%7

    first_day = dates[decal*48]
    last_day = dates[-1].date()

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
        if first_day.date()>=last_day or (day_max != -1 and first_day.date()>=day_max):
            break
        arr = np.array(df.loc[first_day:first_day+pd.Timedelta('1 day')][:-1].fillna(method="bfill").fillna(method="ffill").transpose().values)
        if np.sum(np.isnan(arr)) >0:
            arr = np.zeros(shape=arr.shape)
        for (d1,d2) in empty_weeks:
            if first_day>=d1 and first_day<d2:
                arr = np.nan*np.zeros(shape=arr.shape)

        if len(result) == 0:
            result = arr
        else:
            result = np.append(result, arr, axis=0)

        first_day+=pd.Timedelta('7 day')
    return result

def create_week_matrix(df, weekday, day_max = -1):
    '''
    Create the matrix where a row corresponds to an entire week, and each column to a time slot
    '''

    #for a monday, we go from tuesday to monday
    weekday+=1

    result=np.array([])
    # we start at week 1 of 2011
    dates = df.index[48:]
    first_day = dates[0].weekday()
    decal = (7-first_day+weekday)

    first_day = dates[decal*48]
    last_day = dates[-1].date()

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
        if (first_day+pd.Timedelta('6 day')).date()>=last_day or (day_max != -1 and (first_day+pd.Timedelta('6 day')).date()>=day_max):
            break
        arr = np.array(df.loc[first_day:first_day+pd.Timedelta('7 day')][:-1].transpose().values)

        if len(result) == 0:
            result = arr
        else:
            result = np.append(result, arr, axis=0)

        first_day+=pd.Timedelta('7 day')
    return result

# def create_week_matrix_from_slots(df, weekday, hour, min):
#     '''
#     Create the matrix where a row corresponds to an entire week, starting at a precise time slot, and each column to a time slot
#     '''
#     result=np.array([])
#     dates = df.index
#     first_day = dates[0].weekday()
#     decal = (7-first_day+weekday)%7
#     print("Decal: "+str(decal*48+2*hour+int(min/30)))
#     first_day = dates[decal*48+2*hour+int(min/30)]
#     last_day = dates[-7*48].date()
#     print(first_day)
#     empty_weeks = [(datetime(year=2012,month=12,day=28),datetime(year=2013,month=1,day=4)),
#                    (datetime(year=2013,month=2,day=2),datetime(year=2013,month=2,day=9)),
#                    (datetime(year=2013,month=3,day=6),datetime(year=2013,month=3,day=13)),
#                    (datetime(year=2013,month=4,day=10),datetime(year=2013,month=4,day=17)),
#                    (datetime(year=2013,month=5,day=13),datetime(year=2013,month=5,day=20)),
#                    (datetime(year=2013,month=6,day=12),datetime(year=2013,month=6,day=19)),
#                    (datetime(year=2013,month=7,day=16),datetime(year=2013,month=7,day=23)),
#                    (datetime(year=2013,month=8,day=15),datetime(year=2013,month=8,day=22)),
#                    (datetime(year=2013,month=9,day=14),datetime(year=2013,month=9,day=21)),
#                    (datetime(year=2013,month=10,day=18),datetime(year=2013,month=10,day=25)),
#                    (datetime(year=2013,month=11,day=20),datetime(year=2013,month=11,day=27)),
#                    (datetime(year=2013,month=12,day=22),datetime(year=2013,month=12,day=29))]
#
#     while True:
#         if (first_day+pd.Timedelta('7 day')).date()>last_day:
#             break
#         arr = np.array(df.loc[first_day:first_day+pd.Timedelta('7 day')][:-1].fillna(method="bfill").fillna(method="ffill").transpose().values)
#         if np.sum(np.isnan(arr)) >0:
#             arr = np.zeros(shape=arr.shape)
#         for (d1,d2) in empty_weeks:
#             if (first_day>=d1 and first_day<d2) or (first_day+pd.Timedelta('7 day')>=d1 and first_day+pd.Timedelta('7 day')<d2):
#                 arr = np.nan*np.zeros(shape=arr.shape)
#
#         if len(result) == 0:
#             result = arr
#         else:
#             result = np.append(result, arr, axis=0)
#
#         first_day+=pd.Timedelta('7 day')
#     return result
