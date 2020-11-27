import os
import time
import pandas as pd
import datetime as dt
import numpy as np
from datetime import datetime

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=300"
PAIR_LIST = ["USDT_BTC"]
DATA_DIR = "data"
COLUMNS = ["date","high","low","open","close","volume","quoteVolume","weightedAverage"]

def get_data(pair):
    datafile = os.path.join(DATA_DIR, pair+".csv")
    timefile = os.path.join(DATA_DIR, pair)
    endDate = dt.datetime(2020,10,6,17,20)
    end_time = datetime.timestamp(endDate)
    if os.path.exists(datafile):
        newfile = False
        start_time = int(open(timefile).readline()) + 1
    else:
        newfile = True
        startDate = dt.datetime(2016,10,6,16,20)
        #startDate = dt.datetime(2020,6,30)
        #endDate = dt.datetime(2019,6,28)
        start_time = datetime.timestamp(startDate)
        spl = dt.datetime(2017,1,6,17,20)
        splt = datetime.timestamp(spl)
        print("3 Months: ")
        print(splt-start_time)
        #end_time = datetime.timestamp(endDate)
        #start_time = 1388534400     # 2014.01.01
        #end_time = 9999999999#start_time + 86400*30
    print("Get %s from %d to %d" % (pair, start_time, end_time))
    split_time = start_time+7948800
    url = FETCH_URL % (pair, start_time, start_time+7948800)
    print(url)
    df = pd.read_json(url, convert_dates=False)
    print('New Indices: '+str(dt.datetime.fromtimestamp(start_time)))
    print('New Indices: '+str(dt.datetime.fromtimestamp(split_time)))
    #print(type(start_time))
    for i in range(1, 15):
        #print('Indices: '+str(split_time)+", "+str(i))
        url = FETCH_URL % (pair, split_time, split_time+7948800)
        #print(url)
        df_split = pd.read_json(url, convert_dates=False)
        df = df.merge(df_split, how = "outer")
        split_time = split_time+7948800
        print('New Date: '+str(dt.datetime.fromtimestamp(split_time)))
    print('FINISHED CYCLE:')
    url = FETCH_URL % (pair, split_time, end_time)
    df_split = pd.read_json(url, convert_dates=False)
    df = df.merge(df_split, how = "outer")
    print(len(df))
    #url = FETCH_URL % (pair, start_time, end_time)
    #print(url)

    #df = pd.read_json(url, convert_dates=False)

    #import pdb;pdb.set_trace()

    if df["date"].iloc[-1] == 0:
        print("No data.")
        return

    end_time = df["date"].iloc[-1]
    ft = open(timefile,"w")
    ft.write("%d\n" % end_time)
    ft.close()
    outf = open(datafile, "a")
    if newfile:
        df.to_csv(outf, index=False)
    else:
        df.to_csv(outf, index=False, header=False)
    outf.close()
    print("Finish getting data from poloniex.")
    time.sleep(30)

def get_data_from_poloniex():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
    pairs = [pair for pair in df.columns if pair.startswith('USDT')]
    print(pairs)
    for pair in pairs:
      if pair.startswith('USDT_BTC') and pair.endswith('BTC'):
        get_data(pair)
      time.sleep(2)


def main():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
    pairs = [pair for pair in df.columns if pair.startswith('BTC')]
    print(pairs)
    for pair in pairs:
      if pair.startswith('BTC_ETH') and pair.endswith('ETH'):
        get_data(pair)
      time.sleep(2)
    #for pair in pairs:
        #get_data(pair)
       #time.sleep(2)

if __name__ == '__main__':
    main()
