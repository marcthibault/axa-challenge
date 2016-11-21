import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from datetime import datetime
from csv_reader_by_line import *

rdr = csv_reader_by_line("csv/Gestion.csv", ";")
header = rdr.header

class Intraday_volume_rate:
    def __init__(self, w_hol=False, w_ds=False):
        self.d = [np.array([]) for i in range(0,48)]
        self.w_hol = w_hol
        self.w_ds = w_ds

    def add_data(self, time, v, hol = False, ds = False):
        l = time.split(" ")
        d = l[0].split("-")
        t = l[1].split(":")

        ind = int(t[0])*2+int(int(t[1])/30)

        if (hol and not w_hol) or (ds and not w_ds):
            return
        else:
            self.d[ind] = np.append(self.d[ind], int(v))

    def return_mean(self):
        return [np.mean(e) for e in self.d]

    def return_var(self):
        return [np.var(e) for e in self.d]

results = {}
c_time = header.index("DATE")
c_vol = header.index("CSPL_RECEIVED_CALLS")

print("Start of processing")
counter = 0
step = int(1e5)
while True:
    if counter%step==0 and counter > 0:
        print("Processed entries :"+str(counter))
    try:
        item = rdr.next_line()
        l = item[c_time].split(" ")
        d = l[0].split("-")
        t = l[1].split(":")
        time = datetime(year=int(d[0]), month=int(d[1]), day=int(d[2]))

        if time.weekday() not in results:
            results[time.weekday()] = Intraday_volume_rate()
        results[time.weekday()].add_data(item[c_time], item[c_vol])

    except StopIteration:
        break
    counter+=1

print("Process finished, now printing...")

times = []
for e in range(0,48):
    times += [e/2]

for e in results:
    plt.plot(times,results[e].return_mean(), label=e)

plt.legend()
plt.show()

for e in results:
    plt.plot(times,results[e].return_var(), label=e)

plt.legend()
plt.show()

print("Study finished!")
