import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from datetime import datetime
from rf_estim import *
from gb_366_estim import *
from compute_missing_ar import *

def load_data_create_days(csv_data, day_max=-1):
    results = []
    for i in range(0,7):
        print("Day: "+str(i))
        mat = create_week_matrix(csv_data,i, day_max)
        #results+=[(mat,rf_create_regr(mat))]
        results+=[mat]
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

            if current_cat not in datas:
                print("Predicting " + current_cat)
                # predict_cat(current_cat)
                datas[current_cat] = pd.read_csv("csv/"+current_cat+"_result.csv", header=0, sep=";", parse_dates=True, infer_datetime_format =True, index_col=0).sort_index() * 1.6075

                # datas[current_cat]=create_data(current_cat) # submission
                # datas[current_cat]=create_data_hour(current_cat)

            print("Filling in " + current_cat + " at date " + current_date.strftime("%B %d, %Y"))
            r_h = int(datas[current_cat].xs(current_date))
            #if np.isnan(r_h):
            #    print("pb with "+current_cat)
            result_line = line.replace("\n", "")[:-1] + str(int(np.ceil(r_h)))
            #result_line=line.replace("\n","")[:-1]+str(0)
            f_r.write(result_line + "\n")

        else:
            f_r.write(line)
