import itertools
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

colors = ['#EF476F', '#118AB2', '#06D6A0', '#073B4C', '#FFD166']
lighter_colors = ["#FFACBB", "#33ACD4"]


def init_plot():
    pd.set_option('display.max_columns', 50)
    sns.set_context('talk')
    sns.set_palette(colors)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def marker_settings(edgecolor, markersize=10):
    return dict(marker='o', markerfacecolor='white', markeredgecolor=edgecolor, markersize=markersize, markeredgewidth=2)


def plot_trajectories(ax, trajs, label='Simulation', estimator=np.mean, linestyle='--', color=None, markersize=10):

    if color:
        extra_settings = marker_settings(color)
    else:
        extra_settings = dict(marker='o', markersize=markersize)

    total_years = 20
    sns.lineplot(x=itertools.chain(*[itertools.repeat(i, trajs.shape[1]) for i in range(total_years + 1)]), y=trajs.flatten(), estimator=estimator, label=label, ax=ax, linestyle=linestyle, lw=4, **extra_settings)


def plot_cutoffs(ax, cutoffs, color):
    for x in cutoffs:
        ax.axvline(x, color=color, linestyle='--')

def print_pval(pval):
    if pval < 0.001:
        return 'p<0.001'
    if pval < 0.01:
        return f"p={pval:.03f}"

    return f"p={pval:.02f}"

def wald_interval_95(p, n):
    err = np.sqrt(p * (1 - p) / n)
    return (p - 1.96 * err, p + 1.96 * err)

def count_zero_runs(column):
    return [len(list(v)) for k, v in itertools.groupby(column == 0) if k]


def hazard_from_runlengths(runlengths):

    zerohazard_counts = defaultdict(lambda: defaultdict(int))
    zerohazard = np.zeros(max(runlengths))
    zerohazard_variances = np.zeros(max(runlengths))

    for run in runlengths:
        for i in range(1, run):
            zerohazard_counts[i]['Y'] += 1

        zerohazard_counts[run]['N'] += 1

    for observed, counts in zerohazard_counts.items():
        N = counts['Y'] + counts['N']
        phat = counts['Y'] / N

        zerohazard[observed - 1] = phat
        zerohazard_variances[observed - 1] = phat*(1-phat)/N

    return zerohazard_counts, zerohazard, zerohazard_variances
