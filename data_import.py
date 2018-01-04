import datetime
import pandas as pd
import pandas_datareader
import os
from urllib.request import urlopen
import csv

def getStockData(stock, start, end):
    path = 'data/stocks/' + stock

    f = pandas_datareader.DataReader(stock, 'yahoo', start, end)

    f.to_csv(path + '/open_close.csv')

    return path + '/open_close.csv'


def createStockData(stock):

    start = datetime.date(1995, 1, 1)
    end = datetime.date.today()

    path = 'data/stocks/' + stock

    if os.path.exists(path) == False:
        os.makedirs(path)

    getStockData(stock, start, end)


def updateStockData(stock):
    path = 'data/stocks/' + stock + '/open_close.csv'
    folder_path = 'data/stocks/' + stock

    df = pd.read_csv(path, index_col='Date', parse_dates=True)

    last_line = df.tail(1)

    last_date = last_line.index[0]

    print(last_date)

    last_date_obj = datetime.datetime.strptime(str(last_date).split(" ")[0], '%Y-%m-%d')

    start = last_date_obj + datetime.timedelta(days=1)
    end = datetime.date.today()

    if start == end:
        print ('Real time updates are taking care already')
    else:

        print(start, end)
        try:
            f = pandas_datareader.DataReader(stock, 'yahoo', start, end)
            df_updated = pd.concat([df,f])

            df_updated.to_csv(folder_path + '/open_close.csv')
            print ('Open to Close updated!')

        except Exception as e:
            print ('Nothing new to get')


def checkStockHistoricalData(stock):

    path = 'data/stocks/' + stock + '/open_close.csv'

    if os.path.exists(path):
        updateStockData(stock)
    else:
        createStockData(stock)


def getRealTimeDataFromYahoo(stock):

    try:
        sourceCode = urlopen('http://finance.yahoo.com/q?s='+stock).read()

        date = datetime.datetime.today().strftime("%Y-%m-%d")
        time = datetime.datetime.today().strftime("%H:%M:%S")
        quote = sourceCode.split('<span id="yfs_l84_'+stock.lower()+'">')[1].split('</span>')[0]
        print (quote)
        open = sourceCode.split('Open:</th><td class="yfnc_tabledata1">')[1].split('</td>')[0]
        print (open)
        high = sourceCode.split('<span id="yfs_h53_'+stock.lower()+'">')[1].split('</span>')[0]
        print (high)
        low = sourceCode.split('<span id="yfs_g53_'+stock.lower()+'">')[1].split('</span>')[0]
        print (low)
        volume = sourceCode.split('<span id="yfs_v53_'+stock.lower()+'">')[1].split('</span>')[0]
        print (volume)

        _volume = volume.replace(",", "")

        data = {'Date': [date],
                'Time': [time],
                'Quote': [quote],
                'Open': [open],
                'High': [high],
                'Low': [low],
                'Volume':[_volume]}

    except Exception as e:
        print (str(e))

    try:
        filepath = 'data/stocks/' + stock + '/real_time.csv'


        if os.path.exists(filepath):
            new_minute_quote = pd.DataFrame(data)
            minute_quote = pd.read_csv(filepath)
            merged_df = pd.concat([minute_quote,new_minute_quote])
            merged_df.to_csv(filepath, sep=',', index=False)
            print ('File updated, new quote saved')

        else:
            minute_quote = pd.DataFrame(data)
            print (minute_quote)
            minute_quote.to_csv(filepath, sep=',', index=False)

            print ('File created, first quote saved')

    except Exception as e:
        print (str(e))


ticker = 'AAPL'

checkStockHistoricalData(ticker)

getRealTimeDataFromYahoo(ticker)

