import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from datetime import datetime
from csv_reader_by_line import *
from collections import Counter

f = open('submission.txt', 'r')

ds = set()
ts = []

for line in f:
    line = line.replace("\n","").replace("\t"," ")
    d = line.split(" ")[0]
    if d[0] == '2':
        dd = d.split("-")
        ds.add(datetime(year=int(dd[0]), month=int(dd[1]), day=int(dd[2])).date())
    ts += [" ".join(line.split(" ")[2:-1])]
ld = list(ds)
ld.sort()
nb_days = [0]*7
for d in ld:
    nb_days[d.weekday()]+=1
    print(d)
print(ld[0].isocalendar()[1])
print(nb_days)
print(Counter(ts))
