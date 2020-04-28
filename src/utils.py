def first_date(df):
    date = str(df.head().index.values[-1])
    return date[:10]

def after_date_str(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d").date()
    d2 = datetime.strptime(d2, "%Y-%m-%d").date()
    return d1 > d2

def todays_data(dt):
    data = {}
    for f in os.listdir(DATA_DIR):
        df = pd.read_csv(os.path.join(DATA_DIR, f), index_col=0, parse_dates=True)
        if after_date_str(dt, first_date(df)):
            continue
        df = df.loc[dt:]
        data[f[:len(f)-4]] = df
    return data
