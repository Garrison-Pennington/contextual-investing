import datetime
from functools import partial


from model.market import Market
from model.portfolio import Portfolio
from model.strategy import bbands, bband_mask
from data.data_prep import ma_dev


def sim(strategy, date_range=(None, None), portfolio=None, **kwargs):
    """
    Simulate an investing strategy over a specified date range

    :param strategy: A function that takes a Market object and outputs a list of trades
    :param date_range: (start_date, end_date)
    :type date_range: (datetime.datetime, datetime.datetime)
    :param portfolio: A Portfolio object to start with
    :type portfolio: Portfolio
    :return:
    """
    start_date, end_date = date_range
    portfolio = Portfolio(10000, Market(start_date)) if portfolio is None else portfolio
    end_date = datetime.datetime.today() if end_date is None else end_date
    portfolio.log_trades = True
    portfolio.log_failures = False
    portfolio.log_daily = True
    while portfolio.market.today < end_date:
        for ticker, action in strategy(portfolio.market, **kwargs):
            if action:
                portfolio.buy(ticker)
            else:
                portfolio.sell(ticker)
        portfolio.next_day()
    portfolio.summarize_performance()
    portfolio.graph_value()
    command = input("Finished? Enter to quit, any other key to debug")
    while command:
        command = input()
        if command:
            exec(command)


start, end = datetime.datetime(2000, 1, 25), datetime.datetime(2018, 1, 25)

sim(
    bbands,
    (start, end),
    Portfolio(10000, Market(start, augments=[ma_dev, partial(bband_mask, buy=-2, sell=1)]))
)
