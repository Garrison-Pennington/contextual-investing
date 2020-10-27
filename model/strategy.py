from utils.__init__ import*

class Rule(object):
    """docstring for Rule."""

    def __init__(self, metric_a, metric_b, period, rel, prev_rule=None, c_rule=None, action=1):
        """
        Define a buy or sell rule based on a relationship between two metrics
        occuring over a specified number of days, and possibly after another
        rule has been satisfied
        """
        super(Rule, self).__init__()
        self.metric_a = metric_a
        self.metric_b = metric_b
        self.period = period
        self.criteria = rel
        self.prev_rule = prev_rule
        self.concurrent_rule = c_rule
        self.action = action


    def check(self, df, dt=None):
        if dt is None:  # If no date is specified, use the most recent
            if not len(df.head().index.values):
                return None
            dt = df.head().index.values[0]
        elif pandas_dt_to_date(dt) > pandas_dt_to_date(df.head().index.values[0]):
            print(f"Stock has no data for {dt}")
            return None
        else:
            df = df.loc[dt:]
        # Did we meet previous criteria?
        if self.prev_rule is not None:
            pre = self.prev_rule.check(df.iloc[self.period:])
            if pre is None or not pre:
                return pre
        # Do we meet concurrent criteria?
        if self.concurrent_rule is not None:
            conc = self.concurrent_rule.check(df)
            if conc is None or not conc:
                return conc
        # Do we meet primary criteria over the period?
        for i in range(self.period):
            m_a = df[self.metric_a].iloc[i]
            m_b = df[self.metric_b].iloc[i]
            if not self.criteria(m_a, m_b):
                return False

        # Yes to all
        return True


def average_cross(ma2=50, ma1=200):
    ma1 = f"{ma1} MA"
    ma2 = f"{ma2} MA"
    avg50_under_avg200 = Rule(ma1, ma2, 1, lambda a, b: a < b)
    avg_buy = Rule(ma1, ma2, 1, lambda a, b: a > b, prev_rule=avg50_under_avg200)
    avg50_over_avg200 = Rule(ma1, ma2, 1, lambda a, b: a > b)
    avg_sell = Rule(ma1, ma2, 1, lambda a, b: a < b, prev_rule=avg50_over_avg200, action=-1)

    def inner(df):
        if avg_buy.check(df):
            return avg_buy.action
        if avg_sell.check(df):
            return avg_sell.action
        return 0

    return inner


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
