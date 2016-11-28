import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from datetime import datetime
from submit import *
from knn_estim import *

file = "Services"

df_calls = pd.read_csv("csv/"+file+".csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()

df_calls_estim = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
datas = create_data(file)

for e in df_calls.index:
    if np.isnan(df_calls.loc[e,"CSPL_RECEIVED_CALLS"]):
        weekday = e.weekday()
        w_estim = e.isocalendar()[1] + 51
        if e.year == 2013:
            w_estim+=52
        if np.sum(np.isnan(datas[weekday][0][w_estim-2]))==0:
            print(e)
            #r = k_nn_estim_oneday(datas[weekday][w_estim-2],datas[weekday],5, dist_eucl)
            #r_h = r[2*e.hour+int(e.minute/30)]
            r = datas[weekday][1].predict([datas[weekday][0][w_estim - 2]])
            r_h = np.ceil(r[0][2*e.hour+int(e.minute/30)])

            df_calls_estim.loc[e,"CSPL_RECEIVED_CALLS"]=np.ceil(r_h)

plt.plot(df_calls.loc['20121102':'20131230'].values)
plt.plot(df_calls_estim.loc['20121102':'20131230'].values)
plt.show()
