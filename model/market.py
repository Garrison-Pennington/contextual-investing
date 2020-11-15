import datetime

from data import all_local_data, local_stock_data


class Market:

    def __init__(self, current_date=None, tickers=None, days_before=365, augments=None):
        self.today = datetime.datetime.today() - datetime.timedelta(days=days_before) if current_date is None else current_date
        self.start_date = self.today
        self.data = all_local_data() if tickers is None else {t.upper(): local_stock_data(t.upper()) for t in tickers}
        for t in self.data:
            if augments is not None:
                tmp = self.data[t].iloc[::-1]
                for aug in augments:
                    tmp = aug(tmp)
                self.data[t] = tmp.iloc[::-1]

    def __getitem__(self, item):
        return self.data[item].loc[self.today:]

    def current_price(self, item):
        data = self[item]
        return (data['open'].iloc[0] + data['close'].iloc[0]) / 2

    def next_day(self):
        self.today += datetime.timedelta(days=1)
