import os
import json
from pathlib import Path
from urllib.request import urlopen

from alpha_vantage.timeseries import TimeSeries

import pandas as pd

DATA_DIR = Path.expanduser(Path("~/src/contextual-investing/.data/"))
IDX_DIR = DATA_DIR.joinpath('indices/')
STOCK_DIR = DATA_DIR.joinpath('stocks/')
TWITTER_DIR = DATA_DIR.joinpath('twitter/')


def all_local_data():
    return {t[:-4]: local_stock_data(t[:-4]) for t in os.listdir(STOCK_DIR)}


def local_stock_data(ticker):
    base_dir = STOCK_DIR.joinpath(f"{ticker.upper()}")
    if base_dir.joinpath("price.csv").exists():
        return pd.read_csv(base_dir.joinpath("price.csv"), index_col=0, parse_dates=True)
    df = historical_data(ticker.upper())
    if not base_dir.exists():
        os.mkdir(base_dir)
    df.to_csv(base_dir.joinpath("price.csv"))
    return df


def historical_data(ticker):
    data, metadata = TimeSeries(key='6LHMANWWZ2Y7DA05', output_format='pandas').get_daily(symbol=ticker, outputsize='full')
    return data.rename(columns={k: k[3:] for k in data})  # Headers come with a 'n. ' prefix that's not necessary


def historical_financials(ticker, statement, quarterly=True, limit=400):
    """

    :param ticker:
    :type ticker: str
    :param statement: one of ('income', 'balance-sheet', 'cash-flow')
    :type statement: str
    :param quarterly: True for quarterly statements, False for annual
    :param limit:
    :return:
    """
    base_url = "https://financialmodelingprep.com/api/v3/"
    statement = statement.lower()
    if all(v in statement for v in ("cash", "flow")):
        statement = "cash-flow-statement/"
    elif all(v in statement for v in ("balance", "sheet")):
        statement = "balance-sheet-statement/"
    elif statement == "income":
        statement = "income-statement/"
    else:
        raise ValueError("statement must be one of ('income', 'balance-sheet', 'cash-flow')")
    period = "period=quarter" if quarterly else ""
    url = base_url + statement + ticker.upper() + "?" + period + f"&limit={limit}&apikey=772d9d9a243884a642ce5eb8227a747b"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)


def local_financials(ticker, statement, quarterly=True, **kwargs):
    statement = statement.lower()
    if all(v in statement for v in ("cash", "flow")):
        statement = "cash_flow"
    elif all(v in statement for v in ("balance", "sheet")):
        statement = "balance_sheet"
    elif statement == "income":
        statement = "income"
    else:
        raise ValueError("statement must be one of ('income', 'balance-sheet', 'cash-flow')")
    statement = ("quarterly_" if quarterly else "annual_") + statement
    base_dir = STOCK_DIR.joinpath(f"{ticker.upper()}")
    fp = base_dir.joinpath(f"{statement}.json")
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    if not base_dir.exists():
        os.mkdir(base_dir)
    data = historical_financials(ticker, statement, quarterly, **kwargs)
    with open(fp, "w+") as f:
        json.dump(data, f)
    return data
