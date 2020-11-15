import csv, time, os

from data import *

nasdaq = dict()

with open('/home/noisette/src/contextual-investing/.data/indices/companylist.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)
    for r in reader:
        if r[3] != 'n/a':
            cap = float(r[3][1:-1]) * (10 if r[3][-1] == "B" else 1)
            nasdaq[r[0]] = cap

nasdaq = {k: v for k, v in reversed(sorted(nasdaq.items(), key=lambda item: item[1]))}

tickers = [k for k, v in nasdaq.items()]
owned = [f[:-4] for f in os.listdir('/home/noisette/src/contextual-investing/.data/')]
tickers = filter(lambda t: t not in owned, tickers)

i = 0

while i < 500:  # Max 500 calls per day
    try:
        t = next(tickers)
        print(f"fetching latest data on {t}")
        i += 1
        local_stock_data(t)
        if not i % 5:
            time.sleep(60)  # Max 5 calls per min
    except ValueError as e:
        print("Value error", e)
