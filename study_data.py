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
    estims_d_gb = []
    estims_w_rf = []
    estims_w_gb = []
    for i in range(0,336):
        #estims_d_ab+=[Pipeline([("reduce_dim", PCA(n_components=40)),
        #                 ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), loss="exponential", learning_rate=1.0))])]
        #estims_w_ab+=[Pipeline([("reduce_dim", PCA(n_components=100)),
        #                 ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), loss="exponential", learning_rate=1.0))])]
        estims_d_rf += [Pipeline([("reduce_dim", PCA(n_components=10)),
                                  ("regr", RandomForestRegressor(n_estimators=50, max_depth=2))])]
        estims_w_rf += [Pipeline([("reduce_dim", PCA(n_components=100)),
                                  ("regr", RandomForestRegressor(n_estimators=50, max_depth=5))])]
        estims_d_gb += [Pipeline([("reduce_dim", PCA(n_components=10)),
                                  ("regr", GradientBoostingRegressor(max_depth=2))])]
        estims_w_gb += [Pipeline([("reduce_dim", PCA(n_components=100)),
                                  ("regr", GradientBoostingRegressor(max_depth=5))])]

    #return [estims_d_ab, estims_w_ab]
    #return [estims_d_ab, estims_w_ab, estims_d_rf, estims_w_rf, estims_d_gb, estims_w_gb]
    #return [estims_d_rf, estims_w_rf, estims_d_gb, estims_w_gb]
    return [estims_d_rf, estims_w_rf]


file = "Téléphonie"

df_calls = pd.read_csv("csv/"+file+".csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()

last_call = 0
for e in df_calls.index:
    if np.isnan(df_calls.loc[e,"CSPL_RECEIVED_CALLS"]):
        to_assess = False
        for (d1,d2) in empty_weeks:
            if d2 > e >= d1:
                to_assess = True
                break

        if not to_assess:
            df_calls.loc[e, "CSPL_RECEIVED_CALLS"] = last_call

    else:
        last_call = int(df_calls.loc[e,"CSPL_RECEIVED_CALLS"])

df_calls_estim_w_rf = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
#df_calls_estim_w_ab = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
df_calls_estim_w_gb = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
df_calls_estim_d_rf = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
df_calls_estim_d_gb = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
#df_calls_estim_d_ab = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
df_calls_estim_dummy = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
df_calls_estim_dummy_mean = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)
df_calls_estim_total = pd.DataFrame(index=df_calls.index, columns=df_calls.columns)

global_data_w = []
global_data_d = []
local_data_w = [0]*7
local_data_d = [0]*7

for i in range(0,7):
    global_data_w+=[create_week_matrix(df_calls,i)]
    global_data_d+=[create_day_matrix(df_calls,i)]

need_refit = True

regrs = load_estims()



means_call = np.nanmean(global_data_w[6][:90],axis=0)

k_mean = 2
rolling_mean_global = [0]*7
for s in range(0,7):
    rolling_mean_global[s] = np.zeros(global_data_w[s][k_mean-1:].shape)
    for i in range(0,k_mean-1):
        rolling_mean_global[s]+=global_data_w[s][i:-k_mean+i+1]/k_mean
    rolling_mean_global[s]+=global_data_w[s][k_mean-1:]/k_mean

rolling_var_global = [0]*7
for s in range(0,7):
    rolling_var_global[s] = np.zeros(global_data_w[s][k_mean-1:].shape)
    nb_added = 0
    for i in range(0,k_mean-1):

        rolling_var_global[s]+=(global_data_w[s][i:-k_mean+i+1]-rolling_mean_global[s])**2/k_mean
    rolling_var_global[s]+=(global_data_w[s][k_mean-1:]-rolling_mean_global[s])**2/k_mean


plt.plot((global_data_w[0][k_mean-1:103,-48+2*11]-rolling_mean_global[0][:103-k_mean+1,-48+2*11])/rolling_var_global[0][:103-k_mean+1,-48+2*11])
plt.show()

if True:
    for i in range(0, 7):
        local_data_w[i] = global_data_w[i][k_mean-1:97]
        local_data_d[i] = global_data_d[i][k_mean-1:97]
        for h in range(0, 24):
            for minute in [0, 30]:
                print(48 * i + 2 * h + int(minute / 30))
                for est in range(0, len(regrs)):
                    if est % 2 == 0:
                        X,y = create_training_xy_hour(local_data_d[i], h, minute)
                        #X = (X[1:]-X[:-1])
                        #y = (y[1:]-y[:-1])
                        X = X - rolling_mean_global[i][:97-k_mean,-48:]
                        y = y - rolling_mean_global[i][:97-k_mean,-48+2*h+int(minute/30)]
                        regrs[est][48 * i + 2 * h + int(minute / 30)].fit(X,y)
                    else:
                        X, y = create_training_xy_hour(local_data_w[i], h, minute)
                        #X = (X[1:] - X[:-1])
                        #y = (y[1:] - y[:-1])
                        X = X - rolling_mean_global[i][:97-k_mean]
                        y = y - rolling_mean_global[i][:97-k_mean, -48+2*h+int(minute/30)]
                        regrs[est][48 * i + 2 * h + int(minute / 30)].fit(X,y)


for e in df_calls.index:
    if np.isnan(df_calls.loc[e,"CSPL_RECEIVED_CALLS"]):
        print(e)
        weekday = e.weekday()

        w_estim = e.isocalendar()[1] + 51
        if e.isocalendar()[0]==2013:
            w_estim+=52

        to_assess = False
        for (d1,d2) in empty_weeks:
            if e>=d1 and e<d2:
                to_assess = True

                break

        if not to_assess:
            df_calls.loc[e, "CSPL_RECEIVED_CALLS"] = last_call
        else:
            day_data = global_data_d[weekday][w_estim-1]
            print("daydata: "+str(np.sum(np.isnan(day_data))))
            week_data = global_data_w[weekday][w_estim-2]
            rolling_week_data = rolling_mean_global[weekday][w_estim-2-k_mean+1]
            rolling_week_std_data = np.sqrt(rolling_var_global[weekday][w_estim-2-k_mean+1])
            print("rolling_week: " + str(np.sum(np.isnan(rolling_week_data))))
            print("rolling_week: " + str(np.sum(rolling_week_std_data==0)))
            #df_calls_estim_w.loc[e, "CSPL_RECEIVED_CALLS"] = regrs.predict()
            df_calls_estim_dummy.loc[e, "CSPL_RECEIVED_CALLS"] = day_data[2*e.hour+int(e.minute/30)]
            #df_calls_estim_dummy_mean.loc[e, "CSPL_RECEIVED_CALLS"] = means_call[48*weekday+2*e.hour+int(e.minute/30)]
            # if day_data[2*e.hour+int(e.minute/30)] != 0 and means_call[48*weekday+2*e.hour+int(e.minute/30)] != 0:
            #     #df_calls_estim_d_ab.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #     v1 = df_calls_estim_d_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #     #df_calls_estim_d_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data])[0]
            #     v2 = df_calls_estim_d_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[2][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #     #df_calls_estim_d_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[2][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data])[0]
            #     #df_calls_estim_w_ab.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[1][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #     v3 = df_calls_estim_w_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[1][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #     #df_calls_estim_w_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[1][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]
            #     v4 = df_calls_estim_w_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[3][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #     #df_calls_estim_w_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[3][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]
            # else:
            #     #df_calls_estim_d_ab.loc[e, "CSPL_RECEIVED_CALLS"] = 0
            #     v1 = df_calls_estim_d_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data])[0]
            #     v2 = df_calls_estim_d_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[2][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data])[0]
            #     #df_calls_estim_w_ab.loc[e, "CSPL_RECEIVED_CALLS"] = 0
            #     v3 = df_calls_estim_w_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[1][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]
            #     v4 = df_calls_estim_w_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[3][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]

            #v1 = df_calls_estim_d_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data])[0]+day_data[2*e.hour+int(e.minute/30)]
            data_used = (day_data - rolling_week_data[-48:])
            data_used[rolling_week_std_data[-48:]!=0] /= rolling_week_std_data[-48:][rolling_week_std_data[-48:]!=0]
            v1 = df_calls_estim_d_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([data_used])[0]*rolling_week_std_data[-48+2*e.hour+int(e.minute/30)]+rolling_week_data[-48+2*e.hour+int(e.minute/30)]
            #v2 = df_calls_estim_d_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[2][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data])[0]+day_data[2*e.hour+int(e.minute/30)]
            #v2 = df_calls_estim_d_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[2][48*weekday+2*e.hour+int(e.minute/30)].predict([day_data-rolling_week_data[-48:]])[0]+rolling_week_data[-48+2*e.hour+int(e.minute/30)]
            #v3 = df_calls_estim_w_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[1][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]+day_data[2*e.hour+int(e.minute/30)]
            data_used = (week_data - rolling_week_data)
            data_used[rolling_week_std_data != 0] /= rolling_week_std_data[rolling_week_std_data != 0]
            v3 = df_calls_estim_w_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[1][48*weekday+2*e.hour+int(e.minute/30)].predict([data_used])[0]*rolling_week_std_data[-48+2*e.hour+int(e.minute/30)]+rolling_week_data[-48+2*e.hour+int(e.minute/30)]
            #v4 = df_calls_estim_w_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[3][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]+day_data[2*e.hour+int(e.minute/30)]
            #v4 = df_calls_estim_w_gb.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[3][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data-rolling_week_data])[0]+rolling_week_data[-48+2*e.hour+int(e.minute/30)]

            #df_calls_estim_total.loc[e, "CSPL_RECEIVED_CALLS"] = np.max([v1,v2,v3,v4])

            #df_calls_estim_w_rf.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[3][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data/day_data[2*e.hour+int(e.minute/30)]*means_call[48*weekday+2*e.hour+int(e.minute/30)]])[0]*day_data[2*e.hour+int(e.minute/30)]/means_call[48*weekday+2*e.hour+int(e.minute/30)]
            #df_calls_estim_w.loc[e, "CSPL_RECEIVED_CALLS"] = regrs[0][48*weekday+2*e.hour+int(e.minute/30)].predict([week_data])[0]
        #if np.sum(np.isnan(datas[weekday][0][w_estim-2]))==0 and False:
        #    print(e)
            #r = k_nn_estim_oneday(datas[weekday][w_estim-2],datas[weekday],5, dist_eucl)
            #r_h = r[2*e.hour+int(e.minute/30)]
        #    r = datas[weekday][1].predict([datas[weekday][0][w_estim - 2][:48]])
            #r = datas[weekday+2*e.hour+int(e.minute/30)][1].predict([datas[weekday+2*e.hour+int(e.minute/30)][0][w_estim - 2]])
        #    r_h = np.ceil(r[0][2 * e.hour + int(e.minute / 30)])
            #r_h = r[0]
        #    df_calls_estim.loc[e,"CSPL_RECEIVED_CALLS"]=np.ceil(r_h)
    else:
        last_call = int(df_calls.loc[e,"CSPL_RECEIVED_CALLS"])
        df_calls_estim_total.loc[e, "CSPL_RECEIVED_CALLS"] = last_call
        need_refit = False


df_calls_estim_total.to_csv("csv/"+file+"_result.csv", sep=";", header=True, index_label="DATE")

plt.plot(df_calls.values)
plt.plot(df_calls_estim_d_rf.values, linewidth=1., color="turquoise", ls="--", label="d_rf")
#plt.plot(df_calls_estim_d_ab.values, linewidth=1., color="royalblue", ls="--", label="d_ab")
plt.plot(df_calls_estim_w_rf.values, linewidth=1., color="turquoise", ls="-.", label="w_rf")
#plt.plot(df_calls_estim_w_ab.values, linewidth=1., color="royalblue", ls="-.", label="w_ab")
plt.plot(df_calls_estim_d_gb.values, linewidth=1., color="coral", ls="-.", label="w_ab")
plt.plot(df_calls_estim_w_gb.values, linewidth=1., color="coral", ls="-.", label="w_ab")
#plt.plot(np.maximum(df_calls_estim_w_gb.values, np.maximum(df_calls_estim_d_gb.values,np.maximum(df_calls_estim_d_ab.values,np.maximum(df_calls_estim_w_ab.values,np.maximum(df_calls_estim_d_rf.values,df_calls_estim_w_rf.values))))), linewidth=1.5, color="deepskyblue", label="max")
plt.plot(df_calls_estim_dummy.values, linewidth=1.5, color="yellowgreen", label="d-1")
#plt.plot(df_calls_estim_dummy_mean.values, linewidth=1.5, color="chartreuse", label="dm-1")
plt.legend()
plt.show()
