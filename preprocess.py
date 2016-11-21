import numpy as np
import pandas as pd
import sklearn as sk

from datetime import datetime
from csv_reader_by_line import *

rdr = csv_reader_by_line("train_2011_2012_2013.csv", ";")
#rdr.save_first_lines(n=100000)
#rdr = csv_reader_by_line("firstrows.csv", ";")
header = rdr.header
print(header)

c_ass_assign = header.index('ASS_ASSIGNMENT')
c_rcvd_calls = header.index("CSPL_RECEIVED_CALLS")
c_date = header.index("DATE")
header = ["DATE","ASS_ASSIGNMENT","CSPL_RECEIVED_CALLS"]

def process_line(r, item):
    item = [item[c_date]]+[item[c_ass_assign]]+[item[c_rcvd_calls]]
    #item[0] = item[0].split(",")[1]
    if item[1] in r:
        r[item[1]]["nb"] += 1
        r[item[1]]['DATE'].add(item[0])
        r[item[1]]['CSPL_RECEIVED_CALLS'].add(item[2])
    else:
        r[item[c_ass_assign]] = {"nb":1}
        r[item[1]]['DATE'].add(item[0])
        r[item[1]]['CSPL_RECEIVED_CALLS'].add(item[2])

def print_results(r):
    fw = open("output_study.txt","w", newline='\n')
    fw.write("Number of ASS_ASSIGNMENT :"+str(len(r))+"\n")
    fw.write("# List:\n")
    for h in r:
        fw.write("---  "+h+"\n")
        fw.write("Nb of entries :"+str(r[h]["nb"])+"\n")

def add_to_cat_csv(r, item):
    item = [item[c_date]]+[item[c_ass_assign]]+[item[c_rcvd_calls]]
    #item[0] = item[0].split(",")[1]
    if item[1] in r:
        r[item[1]]["csv"].writerow(item)
    else:
        r[item[1]] = {"f":open("csv/"+item[1]+".csv","w")}
        r[item[1]]["csv"] = csv.writer(r[item[1]]["f"],delimiter=";", lineterminator = '\n')
        r[item[1]]["csv"].writerow(header)
        r[item[1]]["csv"].writerow(item)

results = {}
print("Start of processing")
counter = 0
step = int(1e5)
while True:
    if counter%step==0 and counter > 0:
        print("Processed entries :"+str(counter))
    try:
        item = rdr.next_line()
        #process_line(results, item)
        add_to_cat_csv(results, item)
    except StopIteration:
        break
    counter+=1

print("Process finished, now printing...")
#print_results(results)
print("Study finished!")
