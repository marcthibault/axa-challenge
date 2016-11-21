import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from datetime import datetime
from csv_reader_by_line import *


def create_datamatrix(df):
    cols = range(0,48*7)
    index = range(1,103+1)

    results=np.empty((103,48*7))*np.nan

    #df_others=pd.DataFrame(index=range(1,53),columns=cols)

    for index, row in df.iterrows():
        year = index.year
        week = index.week
        weekday = index.weekday()
        hour = index.hour
        minute = index.minute

        if year == 2011:
            results[week-1,48*weekday+2*hour+int(minute/30)]=row["CSPL_RECEIVED_CALLS"]
        elif year == 2012:
            results[week+50,48*weekday+2*hour+int(minute/30)]=row["CSPL_RECEIVED_CALLS"]
        elif year == 2013:
            #df_others[week-1,48*weekday+2*hour+int(minute/30)]=row["CSPL_RECEIVED_CALLS"]
            pass
    print("Missing values :"+str(np.sum(np.isnan(results))))
    return results

def construct_week(df):
    return []

def dist_eucl(x,y):
    return np.mean((np.array(x)-np.array(y))**2)

def dist_eucl2(x,y):
    return np.mean((np.array(x)-np.array(y))**2)+10*(np.sum(np.array(x)<np.array(y)))

def dist_pearson(x,y):
    xa = np.array(x)
    ya = np.array(y)
    mx = np.mean(xa)
    my = np.mean(ya)
    return 1-(len(x)-1)*(np.sum((xa-mx)*(ya-my))/(np.sum((xa-mx)**2)*np.sum((ya-my)**2)))

def k_nn_estim(calls, df, k, dist):

    l_d = []
    l_w = []

    for w in range(0,90):
        w_calls = df[w]

        d = dist(w_calls,calls)
        if len(l_w)<k:
            l_w+=[w]
            l_d+=[d]
        else:
            for i,e in enumerate(l_d):
                if d<=e:
                    l_d = l_d[0:i]+[d]+l_d[i:-1]
                    l_w = l_w[0:i]+[w]+l_w[i:-1]
                    break
    result = np.array(df[l_w[0]+1])
    #result = np.array(df[l_w[0]+1]) - 1/len(calls)*(np.array(df[l_w[0]+1])-np.array(calls))
    for w in l_w[1:]:
        result+=np.array(df[w+1])
        #result+=np.array(df[w+1]) - 1/len(calls)*(np.array(df[w+1])-np.array(calls))
    result= result/len(l_w)
    return result

df_calls = pd.read_csv("csv/Telephonie.csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()
df_calls = df_calls.groupby(df_calls.index).sum()

df_matrix = create_datamatrix(df_calls)
df_matrix = np.nan_to_num(df_matrix)

U,s,V = np.linalg.svd(df_matrix[0:90,:], full_matrices=False)
k_dim = 50
S = np.diag(s[0:k_dim])
df_matrix[0:90,:] = np.dot(U[:,0:k_dim], np.dot(S, V[0:k_dim,:]))

w_estim = 98

for i in range(2,9):
    r = k_nn_estim(df_matrix[w_estim],df_matrix,i, dist_eucl2)
    plt.plot(r, label="estim"+str(i))
plt.plot(df_matrix[w_estim+1], label="True", linewidth=2.0)


plt.legend()
plt.show()
