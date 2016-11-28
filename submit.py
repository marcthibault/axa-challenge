import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from datetime import datetime
from rf_estim import *

def create_data(file):
    print(file)
    df_calls = pd.read_csv("csv/"+file+".csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index()

    results = []
    for i in range(0,7):
        print("Day: "+str(i))
        mat = create_matrix(df_calls,i)
        #mat[:100] = approx_svd(mat[:100], 10)
        results+=[(mat,rf_create_regr(mat))]

    return results

if __name__=="__main__":
    datas = {}
    counter=0
    f = open('submission.txt', 'r',encoding='utf8')
    f_r = open('submission_result.txt', 'w')
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
                if current_cat not in datas:
                    datas[current_cat]=create_data(current_cat)

                weekday = current_date.weekday()
                w_estim = current_date.isocalendar()[1] + 51
                if current_date.year == 2013:
                    w_estim+=52

                #r = k_nn_estim_oneday(datas[current_cat][weekday][w_estim-2],datas[current_cat][weekday],5, dist_eucl)
                r = datas[current_cat][weekday][1].predict([datas[current_cat][weekday][0][w_estim-2]])
                r_h = np.ceil(r[0][2*current_date.hour+int(current_date.minute/30)])

                if np.isnan(r_h):
                    print("pb with "+current_cat)
                result_line=line.replace("\n","")[:-1]+str(int(np.ceil(r_h)))
                f_r.write(result_line+"\n")

        else:
            f_r.write(line)
