import os
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader


def temp_plot_scalar(hz, metric, name=None):
    df = SummaryReader(f'../datasets/runs/{hz}/').scalars
    df = df[df['tag'] == metric]
    print(df)
    train = df.iloc[::2]
    valid = df.iloc[1::2]
    plt.figure()
    ax = plt.gca()
    train.plot(x='step', y='value', kind='line', ax=ax)
    valid.plot(x='step', y='value', kind='line', ax=ax)
    plt.title(f"Model {metric}")
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("Total Number of Epochs", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(f'../datasets/plots/{name}.png', transparent=True)
    plt.close()


def plot_hist(df, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    df.plot(kind='bar')
    ax.tick_params(axis='both', labelsize=15)
    plt.ylabel('Validation accuracy', fontsize=20)
    plt.xlabel('Length of recording (seconds)', fontsize=20)
    plt.xticks(rotation=0)
    plt.ylim([0.4, 1.02])
    plt.savefig(f'../datasets/plots/{name}.png', transparent=True)
    del fig, ax

path = f'../datasets/runs/'
groups = os.listdir(f'../datasets/runs/')
for HZ in groups:
    cur_dir = path + HZ + '/'
    folders = os.listdir(cur_dir)
    vals = {}
    for folder in folders:
        T = folder.split('_')[3][1:]
        file = cur_dir + folder + '/'
        reader = SummaryReader(file).scalars
        vals[T] = reader[reader['tag'] == 'Accuracy: Validation'].iloc[-1]['value']
    vals = pd.Series(vals)
    vals.index = vals.index.astype(int)
    vals = vals.sort_index()
    plot_hist(vals, HZ)
