import numpy as np
import pandas as pd
import sklearn as sk
import csv
from datetime import datetime

class csv_reader_by_line:
    def __init__(self, link, delimiter):
        self.f = open(link, "r")
        self.link = link
        self.reader = csv.reader(self.f, delimiter=delimiter)
        self.n = self.__next__()
        self.header = next(self.n)

    def __next__(self):
        for i,line in enumerate(self.reader):
            yield line

    def next_line(self):
        return next(self.n)

    def save_first_lines(self, n=1000):
        data = pd.read_csv(self.link, nrows=n)
        data.to_csv("firstrows_"+self.link)

if __name__=="__main__":
    rdr = csv_reader_by_line("train_2011_2012_2013.csv", ";")

    t = datetime.now()
    for i in range(0,int(1e1)):
        print(rdr.next_line())
    print(datetime.now()-t)
