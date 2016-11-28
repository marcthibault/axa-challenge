import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from datetime import datetime, date

# def missing_values(data):
#     d = data.copy()
#
#     d_mask = np.isnan(d)
#     k = 20
#     d[d_mask] = 0
#     for i in range(0,20):
#         U,s,V = np.linalg.svd(d)
#         S = np.diag(s[0:k])
#         d_bis = np.dot(U[:,0:k], np.dot(S, V[0:k,:]))
#
#         d[d_mask] = d_bis[d_mask]
#
#     return d
#
#
def dist_eucl(x,y):
     return np.mean((np.array(x)-np.array(y))**2)

def gen_dist_eucl_pen(l):
     def dist_eucl2(x,y):
         return np.mean((np.array(x)-np.array(y))**2)+l*(np.sum(np.array(x)<np.array(y)))
     return dist_eucl2

def dist_pearson(x,y):
     xa = np.array(x)
     ya = np.array(y)
     mx = np.mean(xa)
     my = np.mean(ya)
     return 1-(len(x)-1)*(np.sum((xa-mx)*(ya-my))/(np.sum((xa-mx)**2)*np.sum((ya-my)**2)))

def k_nn_estim_oneday(info_calls, calls, k, dist):
     l_d = []
     l_w = []

     np_info = np.array(info_calls)

     m = np.max(np_info)
     if m != 0:
         np_info=1/m*np_info

     for w in range(0,len(calls)-2):

         w_calls = calls[w]
         if np.max(w_calls) != 0:
             w_calls = 1/np.max(w_calls)*w_calls
         d = dist(w_calls,np_info)

         if np.sum(np.isnan(calls[w,0])) + np.sum(np.isnan(calls[w+1,0])) + np.sum(np.isnan(calls[w+2,:48]))==0:
             if len(l_w)<k:
                 l_w+=[w]
                 l_d+=[d]
             else:
                 for i,e in enumerate(l_d):
                     if d<=e:
                         l_d = l_d[0:i]+[d]+l_d[i:-1]
                         l_w = l_w[0:i]+[w]+l_w[i:-1]
                         break
     #result = 1/l_d[0]*np.array(df[l_w[0]+2,48*weekday:48*(weekday+1)])
     if np.max(calls[l_w[0]+2,:48])!=0:
         result = np.array(calls[l_w[0]+2,:48])/np.max(calls[l_w[0]+2,:48])
     else:
         result = np.zeros(48)
     #result = np.array(df[l_w[0]+1]) - 1/len(calls)*(np.array(df[l_w[0]+1])-np.array(calls))
     for i,w in enumerate(l_w[1:]):
         #result+=1/max(l_d[i+1],1e-7)*np.array(df[w+2,48*weekday:48*(weekday+1)])
         if np.max(calls[w+2,:48]) != 0:
             result+=np.array(calls[w+2,:48])/np.max(calls[w+2,:48])
         #result+=np.array(df[w+1]) - 1/len(calls)*(np.array(df[w+1])-np.array(calls))
     #result= result/np.sum(1/np.array(l_d))
     result= result/len(l_d)*m
     return result


if __name__=="__main__":
    df_calls = pd.read_csv("csv/Téléphonie.csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()
    calls_monday = create_matrix(df_calls,0)
    calls_monday_bis = create_matrix(df_calls,0)
    calls_monday_bis[0:40] = approx_svd(calls_monday[0:40], 5)

    start_date = date(year=2013, month=7, day=19)
    w_estim = start_date.isocalendar()[1]+103

    pd.DataFrame(data=calls_monday).to_csv("test.csv")
    plt.plot(k_nn_estim_oneday(calls_monday[w_estim-2], calls_monday_bis, 4, dist_eucl), label="estim_pen")
    #plt.plot(k_nn_estim_oneday(calls_monday[48], calls_monday, 4, dist_eucl), label="estim")
    plt.plot(calls_monday[w_estim-1,:48], label="true")

    plt.legend()
    plt.show()
