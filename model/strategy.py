def bbands(market):
    orders = []
    for i, t in enumerate(market.data):
        if not len(market[t]):
            continue
        v = market[t]["ma_dev_open"].iloc[0]

        if v > 0:
            orders.append((t, 1))
        elif v < 0:
            orders.append((t, 0))
    return orders


def bband_mask(df, buy=-2, sell=2, buy_below=True, sell_above=True):
    """

    :param df:
    :type df: pandas.DataFrame
    :param buy:
    :param sell:
    :param buy_below:
    :param sell_above:
    :return:
    """
    if buy_below:
        b = df.mask(df < buy, 1)
    else:
        b = df.mask(df > buy, 1)
    if sell_above:
        s = df.mask(df > sell, -1)
    else:
        s = df.mask(df < sell, -1)
    b = b.mask(b != 1, 0)
    s = s.mask(s != -1, 0)
    b = b.mask(b.eq(b.shift(1)), 0)
    s = s.mask(s.eq(s.shift(1)), 0)
    mask = b + s
    # plt.plot(df["ma_dev_open"], label="dev")
    # plt.plot(df["ma_dev_open"], label="action")
    # plt.legend()
    # plt.show()
    df["ma_dev_open"] = mask["ma_dev_open"].astype(float)
    df["ma_dev_close"] = mask["ma_dev_close"].astype(float)
    return df


def instant_bband(df, buy=-2, sell=2, buy_below=True, sell_above=True):
    """

    :param df:
    :type df: pandas.DataFrame
    :param buy:
    :param sell:
    :param buy_below:
    :param sell_above:
    :return:
    """
    if buy_below:
        b = df.mask(df < buy, 1)
    else:
        b = df.mask(df > buy, 1)
    if sell_above:
        s = df.mask(df > sell, 1)
    else:
        s = df.mask(df < sell, 1)
    b = b.mask(b != 1, 0)
    s = s.mask(s != 1, 0)
    b = b.mask(b.eq(b.shift(1)), 0)['ma_dev_open']
    s = s.mask(s.eq(s.shift(1)), 0)['ma_dev_open']
    prices = (df['close'] + df['open']) / 2
    cost = (b * prices).sum()
    rev = (s * prices).sum()
    # plt.plot(df["ma_dev_open"], label="dev")
    # plt.plot(df["ma_dev_open"], label="action")
    # plt.legend()
    # plt.show()
    return cost, rev

