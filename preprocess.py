import numpy as np
import pandas as pd
import sklearn as sk
from datetime import datetime


csvs = {}
chunksize = int(1e6)
counter = 0
index = list(pd.date_range(start="20110101", end="20140101", freq="30MIN")[:-1])
for chunk in pd.read_csv("train_2011_2012_2013.csv", header=0, parse_dates=True, infer_datetime_format =True, index_col=0, sep=";", encoding="utf-8", chunksize=chunksize):
    for e_chunk in chunk.groupby("ASS_ASSIGNMENT"):
        if not e_chunk[0] in csvs:
            temp=e_chunk[1]["CSPL_RECEIVED_CALLS"].groupby(e_chunk[1].index).sum()
            csvs[e_chunk[0]]=pd.DataFrame(index=index, columns=["CSPL_RECEIVED_CALLS"])
            #csvs[e_chunk[0]] = csvs[e_chunk[0]].append(e_chunk[1]["CSPL_RECEIVED_CALLS"]).groupby(csvs[e_chunk[0]].index).sum()
            #csvs[e_chunk[0]]=e_chunk[1]["CSPL_RECEIVED_CALLS"].groupby(e_chunk[1].index).sum()
            #csvs[e_chunk[0]]=pd.DataFrame(index=index, columns=["CSPL_RECEIVED_CALLS"])
            csvs[e_chunk[0]].loc[temp.index.values,"CSPL_RECEIVED_CALLS"]=temp
        else:
            #csvs[e_chunk[0]] = csvs[e_chunk[0]].append(e_chunk[1]["CSPL_RECEIVED_CALLS"])
            temp=e_chunk[1]["CSPL_RECEIVED_CALLS"].groupby(e_chunk[1].index).sum()
            csvs[e_chunk[0]].loc[temp.index.values,"CSPL_RECEIVED_CALLS"]=csvs[e_chunk[0]].loc[temp.index.values]["CSPL_RECEIVED_CALLS"].fillna(0)+temp

    counter+=1
    print(str(counter*chunksize)+" lines processed")

for r in csvs:
    print("Saving "+r)
    csvs[r].groupby(csvs[r].index).sum().to_csv("csv/"+r+".csv", sep=";", header=True, index_label="DATE")
