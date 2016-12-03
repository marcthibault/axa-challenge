import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from datetime import datetime
from submit import *
from knn_estim import *

def load_estims():
    estims_d_ab = []
    estims_w_ab = []
    estims_d_rf = []
    estims_w_rf = []
    for i in range(0,336):
        estims_d_ab+=[Pipeline([("reduce_dim", PCA(n_components=40)),
                         ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), loss="exponential", learning_rate=1.0))])]
        estims_w_ab+=[Pipeline([("reduce_dim", PCA(n_components=100)),
                         ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), loss="exponential", learning_rate=1.0))])]
        estims_d_rf += [Pipeline([("reduce_dim", PCA(n_components=10)),
                                  ("regr", RandomForestRegressor(n_estimators=50, max_depth=2))])]
        estims_w_rf += [Pipeline([("reduce_dim", PCA(n_components=100)),
                                  ("regr", RandomForestRegressor(n_estimators=50, max_depth=5))])]

    return [estims_d_ab, estims_w_ab, estims_d_rf, estims_w_rf]


file = "Téléphonie"

df_calls = pd.read_csv("csv/"+file+".csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()

df_calls_estim = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
#datas = create_data_hour(file)
datas = load_data_create_days(file)

need_refit = True
last_call = 0

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

for e in df_calls.index:
    break
    if np.isnan(df_calls.loc[e,"CSPL_RECEIVED_CALLS"]):
        weekday = e.weekday()
        w_estim = e.isocalendar()[1] + 51
        if e.year == 2013:
            w_estim+=52
        if np.sum(np.isnan(datas[weekday][0][w_estim-2]))==0 and False:
            print(e)
            #r = k_nn_estim_oneday(datas[weekday][w_estim-2],datas[weekday],5, dist_eucl)
            #r_h = r[2*e.hour+int(e.minute/30)]
            r = datas[weekday][1].predict([datas[weekday][0][w_estim - 2][:48]])
            #r = datas[weekday+2*e.hour+int(e.minute/30)][1].predict([datas[weekday+2*e.hour+int(e.minute/30)][0][w_estim - 2]])
            r_h = np.ceil(r[0][2 * e.hour + int(e.minute / 30)])
            #r_h = r[0]
            df_calls_estim.loc[e,"CSPL_RECEIVED_CALLS"]=np.ceil(r_h)

plt.plot(df_calls.values)
#plt.plot(df_calls_estim.loc['20121102':'20131230'].values)
plt.show()
