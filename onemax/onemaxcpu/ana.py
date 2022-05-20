#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot():
    names = ('gen', 'pop', 'elite', 'chromosome', 'tsize', 'mprob', 'elaps')
    df = pd.read_csv("result3.dat", names=names)
    cpu = df[['pop', 'chromosome', 'elaps']].pivot_table(columns='pop',
                                                         index='chromosome',
                                                         values='elaps')[::-1]
    sns.heatmap(cpu, annot=True, square=True)
    plt.show()
