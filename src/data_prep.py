import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os


DATA_DIR = os.path.expanduser('~/dev/contextual-investing/.data/')

# Loaders
def historical_data(ticker):
    ts = TimeSeries(key='6LHMANWWZ2Y7DA05', output_format='pandas')
    data, metadata = ts.get_daily(symbol=ticker, outputsize='full')
    cols = {}
    for k in data:
        cols[k] = k[3:]
    data = data.rename(columns=cols)
    return data


def fix_keys(df):
    new_keys = {}
    for k in df:
        if k[1:3] == '. ':
            new_keys[k] = k[3:]
    return df.rename(columns=new_keys)


def local_stock_data(ticker):
    ticker = ticker.upper()
    dir = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(dir):
        df = pd.read_csv(dir, index_col=0, parse_dates=True)
    else:
        df = historical_data(ticker)
        df.to_csv(dir)
    return df


# Augments
def sma(df, n_days=200):
    if f"{n_days} MA" in df:
        return df
    else:
        df = df.iloc[::-1].copy()
        closes = df['close'].to_numpy()
        ma = np.zeros((len(df), 1))
        for i in range(n_days, len(closes)):
            ma[i] = np.mean(closes[i-n_days:i])
        df[f"{n_days} MA"] = ma
        df = df.iloc[::-1]
        return df


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




def add_ndsma(df, num_days=3000):
    closes = df['close'].to_numpy()[::-1]
    avg = ndsma(closes, num_days)[::-1]
    df[f"{num_days} CMA"] = avg
    return df


def add_percentages(df):
    metrics = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            data = pd.read_csv(os.path.join(dir, filename))
            metrics[filename[:len(filename)-4]] = fn(data)
        else:
            continue
    return metrics
    percents = percentages(df)
    df['% change'] = percents
    return df


def augment_local_file(ticker, augments, augment_args):
    filename = os.path.join(DATA_DIR, f"{ticker}.csv")
    data = pd.read_csv(filename, index_col='date', parse_dates=True)
    for i in range(len(augments)):
        aug = augments[i]
        if not augment_args[i]:
            data = aug(data)
        else:
            data = aug(data, *augment_args[i])
    data.to_csv(filename)
    return data


def augment_all(augments, augment_args):
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            augment_local_file(filename[:len(filename)-4], augments, augment_args)
        else:
            continue
