class Portfolio(object):
    """docstring for portfolio."""

    def __init__(self, principle):
        super(Portfolio, self).__init__()
        self.cash = principle
        self.principle = principle
        self.assets = {}

    def buy(self, ticker, price, count, executed):
        ticker = ticker.upper()
        print(count, price)
        if self.cash - count*price < 0:
            print(f"{ticker} PURCHASE FAILED: Not enough funds")
            return -1
        else:
            self.cash -= count*price
        if ticker in self.assets:
            self.assets[ticker]['count'] += count
            self.assets[ticker]['purchases'].append((count,
                                                     price,
                                                     executed))
        else:
            self.assets[ticker] = {}
            self.assets[ticker]['count'] = count
            self.assets[ticker]['purchases'] = [(count,
                                                 price,
                                                 executed)]
        print(f"Purchased {count} shares of {ticker} for ${price*count} on"
        f" {executed}")
        return 1

    def sell(self, ticker, price, count, executed):
        # Are the shares in the portfolio?
        ticker = ticker.upper()
        if ticker in self.assets:
            if self.assets[ticker]['count'] < count:
                print(f"SALE FAILED: Can't sell {count} shares of {ticker},"""
                f" only have {self.assets[ticker]['count']}")
                return -1
            else:
                # Sell all?
                if count==-1 or count=='all':
                    count = self.assets[ticker]['count']
                self.assets[ticker]['count'] -= count  # Deduct the shares
                # Record the sale
                if 'sell prices' in self.assets[ticker]:
                    self.assets[ticker]['sales'].append((count,
                                                         price,
                                                         executed))
                else:
                    self.assets[ticker]['sales'] = [(count,
                                                     price,
                                                     executed)]
                self.cash += price*count  # Convert to cash
                print(f"Sold {count} shares of {ticker} for {count*price}")
                return 1
        else:
            print(f"SALE FAILED: No shares of {ticker}")
            return -1

    def display_portfolio(self):
        total_value = self.cash
        for t in self.assets:
            stock = self.assets[t]
            last_purchase = stock['purchases'][len(stock['purchases'])-1]
            total_value += stock['count'] * last_purchase[1]
            print(f"{stock['count']} shares of {t} worth a total of"
            f" ${stock['count'] * last_purchase[1]} "
            f"at acquisition")
        print(f"Total portfolio worth ${total_value}")
