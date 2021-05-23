import numpy as np
import pandas as pd
import os

from data import STOCK_DIR

# Loaders


def fix_keys(df):
    new_keys = {}
    for k in df:
        if k[1:3] == '. ':
            new_keys[k] = k[3:]
    return df.rename(columns=new_keys)


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
    for i in range(1, (max_days//step)+1):
        ma += sma(data, i*step)
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


def ma_dev(df, short=50, long=200):
    """

    :param df:
    :type df: pandas.DataFrame
    :param short:
    :param long:
    :return:
    """
    ma = df.rolling(long).mean()
    std = df.rolling(long).std()
    normalized = ((df - ma) / std).rolling(short).mean()
    return df.assign(ma_dev_open=normalized['open'], ma_dev_close=normalized['close'])


def augment_local_file(ticker, augments, augment_args):
    filename = os.path.join(STOCK_DIR, f"{ticker}.csv")
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
    for filename in os.listdir(STOCK_DIR):
        if filename.endswith(".csv"):
            augment_local_file(filename[:len(filename)-4], augments, augment_args)
        else:
            continue
