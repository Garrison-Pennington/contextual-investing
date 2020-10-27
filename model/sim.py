from datetime import timedelta
from model.portfolio import Portfolio
from model.strategy import average_cross
from utils.__init__ import*

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



def sim_day(dt, rules=None, ptf=None):
    data = todays_data(dt)
    if ptf is None:
        ptf = Portfolio(int(input("Enter a number for starting capital")))
    for k in data:
        if dt not in data[k].index:
            continue
        # is the current date on a market break?
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


def sim(start_date, end_date, rules=[average_cross], ptf=None):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    current_date = start_date
    rules = [r() for r in rules]
    while current_date < end_date:
        ptf = sim_day(str(current_date), rules, ptf)
        current_date += timedelta(days=1)
    return ptf


ac = Portfolio(100000)
ac = sim('2015-01-01', '2019-01-01', ptf=ac)
ac.display_portfolio()

# plt.plot(local_stock_data('nvda')["close"])
# plt.show()
