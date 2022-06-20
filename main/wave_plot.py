import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader


def plot_hist(df, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.boxplot(df, labels=df.columns)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(axis='y')
    plt.ylabel('Test accuracy', fontsize=20)
    plt.xlabel('Length of recording T (seconds)', fontsize=20)
    plt.ylim([0.4, 1.02])
    plt.title(f'Test accuracy of 10 runs with {name}HZ (Scaled)', fontsize=28)
    plt.savefig(f'./datasets/plots/{name}.png', transparent=True)
    del fig, ax


def bar_chart():
    path = f'./datasets/runs_full/'
    groups = os.listdir(path)
    for HZ in groups:
        if HZ == 'old_runs':
            continue
        cur_dir = path + HZ + '/'
        folders = os.listdir(cur_dir)
        vals = {}
        for folder in folders:
            T = folder.split('_')[3][1:]
            file = cur_dir + folder + '/'
            reader = SummaryReader(file).text
            vals[T] = reader['value'].apply(lambda x: float(x.split(' ')[3][:-1]))[:10]
        vals = pd.DataFrame(vals)
        vals.columns = vals.columns.astype(int)
        vals = vals.sort_index(axis=1)
        plot_hist(vals, HZ)

def bar_chart2():
    path = f'./datasets/runs_full/'
    groups = os.listdir(path)
    vals = {}
    for HZ in groups:
        if HZ == 'old_runs':
            continue
        cur_dir = path + HZ + '/'
        folders = os.listdir(cur_dir)
        for folder in folders:
            T = folder.split('_')[3][1:]
            file = cur_dir + folder + '/'
            reader = SummaryReader(file).text
            lst = list(reader['value'].apply(lambda x: float(x.split(' ')[3][:-1]))[:10])
            if T not in vals or vals[T] is None:
                vals[T] = []
            vals[T] = vals[T] + lst
    vals = pd.Series(vals)
    vals.index = vals.index.astype(int)
    vals = vals.sort_index()
    plot_hist(vals, 'temp')


def plot_comparison():
    path = f'./datasets/runs/best/'
    groups = os.listdir(path)
    vals = {}
    for i, method in enumerate(groups):
        file = path + method + '/'
        reader = SummaryReader(file).scalars
        vals[i] = reader[reader['tag'] == 'Loss: Validation']['value']
    vals = pd.DataFrame(vals).reset_index().drop(columns=['index'])
    vals = vals.apply(lambda x: np.convolve(x, np.ones(10), 'valid') / 10)
    vals.plot()
    plt.ylabel('Validation loss', fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    # plt.ylim([0.45, 0.905])
    plt.legend([])
    plt.title(f'Validation loss for T10 HZ25 (Scaled)', fontsize=15)
    plt.savefig(f'./datasets/plots/temp.png', transparent=True)

plot_comparison()

def plot_single():
    path = f'./datasets/runs/scaled/10/dataset_10k_scale_T30_H3_HZ10_E1000_PT10_LR0.0001/'
    reader = SummaryReader(path).scalars
    vals = reader[reader['tag'] == 'Accuracy: Validation']['value'].reset_index().drop(columns=['index'])
    vals = vals.apply(lambda x: x[::4])
    vals.plot()
    plt.ylabel('Validation accuracy', fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    plt.savefig(f'./datasets/plots/best.png', transparent=True)
