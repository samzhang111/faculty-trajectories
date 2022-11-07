import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar


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

def enable_matplotlib_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

def disable_matplotlib_latex():
    plt.rcParams.update({
        "text.usetex": False,
    })


def trunclaplace_negloglike(alpha, xs, ms):
    xs = np.array(xs)
    ms = np.array(ms)

    a = np.abs(xs - ms).sum()
    b = np.log(2 - np.exp(-ms/alpha)).sum()

    return len(xs)*np.log(alpha) + a/alpha + b


def fit_trunc_laplace(data, ms):
    result = minimize_scalar(trunclaplace_negloglike, args=(data, ms), bounds=[0, 500], method='Bounded')
    return result.x


def find_mode(vals, bins=50):
    counts, bars = np.histogram(vals, bins=bins)
    i = np.argmax(counts)
    mode = (bars[i] + bars[i+1])/2

    return mode


def plot_laplace_fit(ax, data, label, color, lw=5, linestyle="-"):
    mode = find_mode(data)
    alpha = fit_trunc_laplace(data, mode)
    rv = stats.laplace(loc=mode, scale=alpha)
    x = np.linspace(-25, 25, 100)
    plotted = ax.plot(x, rv.pdf(x), lw=lw, label=f'\\begin{{align*}}{label}\\widehat{{\\alpha}}&={align_and_format(alpha)} \\\\ \\widehat{{\\mu}}&={align_and_format(mode)}\\end{{align*}}', color=color, linestyle=linestyle)
    ax.semilogy()
    return plotted, alpha, mode


def align_and_format(x):
    return f"{x:.2f}"
