import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from datetime import datetime
from utils import *

if __name__=="__main__":
    datas = {}

    f = open('submission.txt', 'r',encoding='utf8')
    f_r = open('submission_result.txt', 'w')
    for line in f:
        nline = line.replace("\n","").replace("\t"," ")
        split_line = nline.split(" ")
        d = split_line[0]

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

                r = k_nn_estim_oneday(datas[current_cat][weekday][w_estim-2],datas[current_cat][weekday],5, dist_eucl)

                r_h = r[2*current_date.hour+int(current_date.minute/30)]
                if np.isnan(r_h):
                    print("pb with "+current_cat)
                result_line=line.replace("\n","")[:-1]+str(int(np.ceil(r_h)))
                f_r.write(result_line+"\n")
        else:
            f_r.write(line)
