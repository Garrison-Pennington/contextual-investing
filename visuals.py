import matplotlib.pyplot as plt
import numpy as np

def show_companies(*argv):
    for arg in argv:
        plt.plot(arg['close'])
    plt.show()


def load_and_show(*argv):
    data = []
    for arg in argv:
        data.append(historical_data_by_ticker(arg))
    show_companies(*data)

def price_v_sma(df):
    df = add_sma(df)
    plt.plot(df['close'])
    plt.plot(df['200 MA'])
    plt.show()
