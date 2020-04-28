from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from portfolio import Portfolio

DATA_DIR = os.path.expanduser('~/dev/investing/.data/')

TICKERS = [
    "TSLA",
    "OTGLY",
    "SNAP",
    "ESPO",
    "VOO",
    "AAPL",
    "AAL",
    "AMD",
    "MMM",
    "NVDA",
    "MSFT"
]

def historical_data(ticker):
    ts = TimeSeries(key='6LHMANWWZ2Y7DA05', output_format='pandas')
    data, metadata = ts.get_daily(symbol=ticker, outputsize='full')
    cols = {}
    for k in data:
        cols[k] = k[3:]
    data = data.rename(columns=cols)
    return data


def local_stock_data(ticker):
    ticker = ticker.upper()
    dir = os.path.join(os.path.expanduser('~/dev/investing/.data/'), f"{ticker}.csv")
    if os.path.exists(dir):
        df = pd.read_csv(dir, index_col=0, parse_dates=True)
    else:
        df = historical_data(ticker)
        df.to_csv(dir)
    return df


def show_company(ticker):
    df = local_stock_data(ticker)
    df['close'].plot()
    plt.show()


def sim_day(dt, rules=None, ptf=None):
    data = todays_data(dt)
    if ptf is None:
        ptf = Portfolio(int(input("Enter a number for starting capital")))
    for k in data:
        if dt not in data[k].index:
            continue
        # is the current date on a market break?
        if data[k].iloc[0][0] == data[k].iloc[1][0]:
            print(f"Market Break on {dt}")
            return ptf
        df = data[k]
        for r in rules:
            action = r(df)
            if action:
                price = df.at[dt, "close"]
            if action >= 1:
                ptf.buy(k, price, int(action), dt)
            elif action <= -1:
                ptf.sell(k, price, int(action), dt)

    return ptf


def first_date(df):
    date = str(df.head().index.values[-1])
    return date[:10]

def after_date_str(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d").date()
    d2 = datetime.strptime(d2, "%Y-%m-%d").date()
    return d1 > d2

def todays_data(dt):
    data = {}
    for f in os.listdir(DATA_DIR):
        df = pd.read_csv(os.path.join(DATA_DIR, f), index_col=0, parse_dates=True)
        if after_date_str(dt, first_date(df)):
            continue
        df = df.loc[dt:]
        data[f[:len(f)-4]] = df
    return data


def average_cross(df, ma1_v=50, ma2_v=200):
    if ma1_v > len(df) or ma2_v > len(df):
        print(f"Not enough days for {max(ma1_v,ma2_v)} average")
        return 0
    else:
        ma1 = sma(df, ma1_v)[f"{ma1_v} MA"]
        ma2 = sma(df, ma2_v)[f"{ma2_v} MA"]
        last_ma_diff = ma1.iloc[1] - ma2.iloc[1]
        current_ma_diff = ma1.iloc[0] - ma2.iloc[0]
        lmad = 1 if last_ma_diff > 0 else -1
        cmad = 1 if current_ma_diff > 0 else -1
        if cmad == lmad:
            return 0
        elif cmad > lmad:
            print(f"{ma1_v} MA broke above {ma2_v} MA")
            return 1
        else:
            print(f"{ma1_v} MA went below {ma2_v} MA")
            return -1


def touch_ma(df, ma=200):
    if ma > len(df):
        print(f"Not enough days for {ma} MA")
        return 0
    else:
        ma_data = sma(df, ma)[f"{ma} MA"]
        prices = df["close"]
        ld = 1 if ma_data.iloc[1] - prices.iloc[1] > 0 else -1
        cd = 1 if ma_data.iloc[0] - prices.iloc[0] > 0 else -1
        if ld==cd:
            print(f"Price hit {ma} MA")
            return 1
        elif cd > ld:
            print(f"Price surpassed {ma} MA")
            return 1
        else:
            print(f"Price fell below {ma} MA")
            return -1


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


def augment_local_file(augment_fn, augment_args, ticker):
    df = local_stock_data(ticker)
    df = augment_fn(df, *augment_args)
    df.to_csv(os.path.join(DATA_DIR, f"{ticker.upper()}.csv"))
    return df


def sim(start_date, end_date, rules=[average_cross, touch_ma], ptf=None):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    current_date = start_date
    while current_date < end_date:
        ptf = sim_day(str(current_date), rules, ptf)
        current_date += timedelta(days=1)
    return ptf


def augment_all(augment_fn, augment_args):
    for f in os.listdir(DATA_DIR):
        ticker = f[:len(f)-4]
        augment_local_file(augment_fn, augment_args, ticker)

# for t in TICKERS:
#     local_stock_data(t)

ac = Portfolio(100000)
ac = sim('2015-01-01', '2016-01-01', ptf=ac)
ac.display_portfolio()
