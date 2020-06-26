from data_prep import local_stock_data
from utils import*

class Market(object):
    """docstring for Market."""

    def __init__(self, date, tickers):
        super(Market, self).__init__()
        self.full_data = {}
        for t in tickers:
            self.full_data[t] = local_stock_data(t)
        self.current_date = date
        self.data = todays_data(current_date)
