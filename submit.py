import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from datetime import datetime
from rf_estim import *
from gb_366_estim import *

def load_data_create_days(file):
    print(file)
    df_calls = pd.read_csv("csv/"+file+".csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()

    results = []
    for i in range(0,7):
        print("Day: "+str(i))
        mat = create_week_matrix(df_calls,i)
        results+=[(mat,rf_create_regr(mat))]

    return results

def load_data_create_slots(file):
    print(file)
    df_calls = pd.read_csv("csv/"+file+".csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()

    results = []
    for i in range(0,7):
        for hour in range(0,24):
            for min in [0,30]:
                print("Day: "+str(i)+" "+str(2*hour+int(min/30)))
                mat = create_week_matrix_from_slots(df_calls,i, hour, min)
                results+=[(mat,gb_create_regr(mat))]

    return results

if __name__=="__main__":
    datas = {}
    counter=0
    f = open('submission.txt', 'r',encoding='utf8')
    f_r = open('submission_result.txt', 'w', encoding='utf8')
    for line in f:
        nline = line.replace("\n","").replace("\t"," ")
        split_line = nline.split(" ")
        d = split_line[0]
        counter+=1
        if(counter%int(1e3)==0):
            print(counter)
        if d[0] == '2':
            dd = d.split("-")
            current_date = datetime(year=int(dd[0]), month=int(dd[1]), day=int(dd[2]), hour=int(split_line[1].split(":")[0]), minute=int(split_line[1].split(":")[1]))
            current_cat = " ".join(split_line[2:-1])

            if current_cat in ['Prestataires']:
                result_line=line.replace("\n","")[:-1]+str(0)
                f_r.write(result_line+"\n")
            else:
                #if current_cat not in datas:
                #    datas[current_cat]=create_data(current_cat)
                    #datas[current_cat]=create_data_hour(current_cat)

                weekday = current_date.weekday()
                w_estim = current_date.isocalendar()[1] + 51
                if current_date.year == 2013:
                    w_estim+=52

                #r = k_nn_estim_oneday(datas[current_cat][weekday][w_estim-2],datas[current_cat][weekday],5, dist_eucl)
                #r = datas[current_cat][weekday][1].predict([datas[current_cat][weekday][0][w_estim-2]])
                #r = datas[current_cat][weekday+2*current_date.hour+int(current_date.minute/30)][1].predict([datas[current_cat][weekday+2*current_date.hour+int(current_date.minute/30)][0][w_estim-2]])
                #r_h = np.ceil(r[0][2*current_date.hour+int(current_date.minute/30)])
                #r_h = np.ceil(r[0])
                if current_date.hour<5 or current_date.hour>23:
                    r_h=100
                elif current_cat == "Téléphonie":
                    if current_date.hour > 9 and current_date.hour < 18:
                        r_h=2000
                    else:
                        r_h=1000
                elif current_cat == "CMS":
                    r_h=1
                elif current_cat == "Nuit":
                    r_h=100
                else:
                    r_h=600
                #if np.isnan(r_h):
                #    print("pb with "+current_cat)
                result_line=line.replace("\n","")[:-1]+str(int(np.ceil(r_h)))
                #result_line=line.replace("\n","")[:-1]+str(0)
                f_r.write(result_line+"\n")

        else:
            f_r.write(line)
