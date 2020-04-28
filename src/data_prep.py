import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os


# Loaders
def historical_data_by_ticker(ticker):
    ts = TimeSeries(key='6LHMANWWZ2Y7DA05', output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')
    data = fix_keys(data)
    return data


def fix_keys(df):
    new_keys = {}
    for k in df:
        if k[1:3] == '. ':
            new_keys[k] = k[3:]
    return df.rename(columns=new_keys)


def local_stock_data(ticker, augments=[add_percentages, add_sma]):
    """
    Loads a local file for a ticker if it exists or downloads it if necessary
    :param ticker: (str) stock ticker to get data for
    :return: Pandas dataframe for the ticker
    """
    ticker = ticker.upper()
    filename = os.path.expanduser(f"~/dev/contextual-investing/.data/{ticker}.csv")
    if os.path.exists(filename):
        data = pd.read_csv(filename, index_col='date', parse_dates=True)
    else:
        data = historical_data_by_ticker(ticker)
        data.to_csv(filename)
    return data


# Augments
def sma(data, num_days):
    data = np.array(data)
    sma = np.zeros_like(data)
    for i in range(num_days, len(data)):
        sma[i] += np.mean(data[i-num_days:i], 0)
    return sma


def ndsma(data, max_days, step=10):
    ma = np.zeros_like(data)
    for i in range(1,(max_days//step)+1):
        ma += sma(data,i*step)
    ma /= (max_days//step)
    return ma


def percentages(df):
    """
    Compute the percentage change between entries filed in reverse chronological order
    :param df: Pandas dataframe containing all data relevant to the ticker
    :return: Numpy array with percentage values such value X represents X%
    """
    closes = df['close'].to_numpy()
    percents = np.zeros_like(closes)
    percents[1:] = np.divide(
                np.subtract(
                    closes[:len(closes)-1],  # Today -> 2nd record
                    closes[1:]),  # Yesterday -> First Record
                closes[1:])*100
    return percents


def add_sma(df, num_days=200):
    closes = df['close'].to_numpy()[::-1]
    avg = sma(closes, num_days)[::-1]
    df[f"{num_days} MA"] = avg
    return df


def add_ndsma(df, num_days=3000):
    closes = df['close'].to_numpy()[::-1]
    avg = ndsma(closes, num_days)[::-1]
    df[f"{num_days} CMA"] = avg
    return df


def add_percentages(df):
    dir = os.path.expanduser('~/dev/investing/.data/')
    metrics = {}
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            data = pd.read_csv(os.path.join(dir, filename))
            metrics[filename[:len(filename)-4]] = fn(data)
        else:
            continue
    return metrics
    percents = percentages(df)
    df['% change'] = percents
    return df


def augment_local_file(ticker, augments):
    filename = os.path.expanduser(f"~/dev/investing/.data/{ticker}.csv")
    data = pd.read_csv(filename, index_col='date', parse_dates=True)
    for aug in augments:
        data = aug(data)
    data.to_csv(filename)
    return data


def augment_all(augments):
    dir = os.path.expanduser('~/dev/investing/.data/')
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            augment_local_file(filename[:len(filename)-4], augments)
        else:
            continue
