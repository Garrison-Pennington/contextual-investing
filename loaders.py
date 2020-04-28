from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from data_prep import local_stock_data


class TimeSeries(object):
    """docstring for TimeSeries."""

    def __init__(self, tickers):
        super(TimeSeries, self).__init__()
        self.stocks = tickers

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, item):
        df = local_stock_data(self.stocks[item])
        data = df.to_numpy()
        return data
