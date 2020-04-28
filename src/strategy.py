def average_cross(df, ma1_v=50, ma2_v=200):
    if ma1_v > len(df) or ma2_v > len(df):
        print(f"Not enough days for {max(ma1_v,ma2_v)} average")
        return 0
    else:
        ma1 = sma(df, ma1_v)[f"{ma1_v} MA"]
        ma2 = sma(df, ma2_v)[f"{ma2_v} MA"]
        last_ma_diff = ma1.iloc[1] - ma2.iloc[1]
        current_ma_diff = ma1.iloc[0] - ma2.iloc[0]
        lmad = 1 if last_ma_diff > 0 else -1
        cmad = 1 if current_ma_diff > 0 else -1
        if cmad == lmad:
            return 0
        elif cmad > lmad:
            print(f"{ma1_v} MA broke above {ma2_v} MA")
            return 1
        else:
            print(f"{ma1_v} MA went below {ma2_v} MA")
            return -1


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
