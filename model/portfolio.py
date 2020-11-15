from collections import defaultdict
import math

import matplotlib.pyplot as plt

from model.market import Market


class Portfolio:

    def __init__(self, principle, market):
        self.principle = principle
        self.cash = principle
        self.holdings = defaultdict(int)
        self.history = []
        self.market = market
        self.value_history = defaultdict(float)
        self.log_trades = False
        self.log_failures = False
        self.log_daily = True
        self.debug = {
            "failed trades": {
                "sells": 0,
                "buys": 0
            },
            "executed": {
                "sells": 0,
                "buys": 0
            }
        }

    def buy(self, ticker):
        price = self.market.current_price(ticker)
        if self.cash >= price:
            self.holdings[ticker] += 1
            self.cash -= price
            self.history.append((ticker, 'bought', price, self.market.today))
            self.debug["executed"]["buys"] += 1
            if self.log_trades:
                print(f"Bought {ticker} at {price}")
        else:
            self.debug["failed trades"]["buys"] += 1
            if self.log_failures:
                print(f"Can't buy {ticker} at ${price} with only ${self.cash} available")

    def sell(self, ticker):
        if self.holdings[ticker] > 0:
            price = self.market.current_price(ticker)
            self.holdings[ticker] -= 1
            self.cash += price
            self.history.append((ticker, 'sold', price, self.market.today))
            self.debug["executed"]["sells"] += 1
            if self.log_trades:
                print(f"Sold {ticker} at {price}")
        else:
            self.debug["failed trades"]["sells"] += 1
            if self.log_failures:
                print(f"Don't own any shares of {ticker} to sell")

    @property
    def value(self):
        return sum([self.market.current_price(t) * q for t, q in self.holdings.items()]) + self.cash

    def next_day(self):
        self.value_history[self.market.today] = self.value
        if self.log_daily:
            print(self.market.today, self.value)
        self.market.next_day()

    def graph_value(self):
        plt.plot(*zip(*self.value_history.items()))
        plt.axhline(y=self.principle)
        plt.show()

    def summarize_performance(self):
        roi = (self.value - self.principle) / self.principle
        annualized = math.pow(1 + roi, 365 / (self.market.today - self.market.start_date).days) - 1
        optimal_return = (max(self.value_history.values()) - self.principle) / self.principle - 1
        print(f"""
Total ROI: {int(roi * 1000) / 10}%
Annualized: {int(annualized * 1000) / 10}%
Total ROI at Peak: {int(optimal_return * 1000) / 10}%
""")

# test = Portfolio(10000, Market(days_before=365*3))
#
# for i in range(600):
#     if random.random() > .5:
#         if random.random() > .5:
#             test.buy("NVDA")
#         else:
#             test.sell("NVDA")
#         print(test.market.today, test.value)
#     test.next_day()
#
# test.graph_value()
