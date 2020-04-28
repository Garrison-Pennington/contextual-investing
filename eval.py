import os
from data_prep import*


def count_days_below_sma(df, num_days=200):
    df = add_sma(df, num_days)
    closes = df['close'].to_numpy()[:200:-1]
    sma = df[f"{num_days} MA"].to_numpy()[:200:-1]
    count = 0
    for i in range(len(closes)):
        if closes[i] < sma[i]:
            count += 1
    return count


def avg_under_sma_run(df, num_days=200):
    df = add_sma(df, num_days)
    closes = df['close'].to_numpy()[:200:-1]
    sma = df[f"{num_days} MA"].to_numpy()[:200:-1]
    count = 0
    on_run = False
    run_length = 0
    for i in range(len(closes)):
        if closes[i] < sma[i]:
            if on_run:
                run_length += 1
            else:
                on_run = True
                count +=1
        else:
            if on_run:
                on_run = False
    return run_length/count


def eval_all_files(fn):
    dir = os.path.expanduser('~/dev/investing/.data/')
    metrics = {}
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            data = pd.read_csv(os.path.join(dir, filename))
            metrics[filename[:len(filename)-4]] = fn(data)
        else:
            continue
    return metrics
