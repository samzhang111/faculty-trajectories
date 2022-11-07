import itertools
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy import stats

from plot_helpers import *

init_plot()

#######################################
# Import data
#######################################

# Empirical trajectories
df_traj_all = pd.read_csv("input/import/all-trajectories.csv")
df_traj_full = pd.read_csv("input/import/full-trajectories.csv")

# Fitted results fits
with open("input/fit/results-all-trajectories.pickle", "rb") as f:
    results_all = pickle.load(f)

mle_cutoffs = results_all['df_reg_scores'].sort_values(by='aic_varying').cutoffs.iloc[0]

# Bootstrapped fits
with open("input/fit/bootstrap-all-trajectories.pickle", "rb") as f:
    bootstrap_results = pickle.load(f)

top_cutoff_1_bootstrap = [x.cutoffs.iloc[0] for x in bootstrap_results['df_reg_scores_bootstrap']]

with open("input/simulate/simulated-all.pickle", "rb") as f:
    simulated_results_all = pickle.load(f)
    sim_trajs_all = simulated_results_all['trajs']

#######################################
# Make model fit plot
#######################################

fig, axes_dict = plt.subplot_mosaic("""
ac
bc
bc
""", figsize=(17, 6))

ax=axes_dict['a']

sns.histplot(list(itertools.chain(*top_cutoff_1_bootstrap)), ax=ax, stat='density', discrete=True, binrange=(0,20), color="gray", edgecolor="gray", alpha=0.2)
sns.kdeplot(list(itertools.chain(*top_cutoff_1_bootstrap)), ax=ax, color="gray")

ax.set_xlim([0, 20])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

ax.set_ylabel("Bootstrapped\nchange points\n")
ax.annotate("A.", (0.02, 0.85), xycoords="axes fraction", fontsize=28)

ax=axes_dict['b']
sns.lineplot(x=df_traj_all.CareerAgeZero, y=df_traj_all.pubs_adj, label='Empirical', ax=ax, lw=4, **marker_settings(colors[0]))
plot_trajectories(ax, sim_trajs_all)

ax.set_xlabel("Year, $t$")
ax.set_ylabel("$q_t$")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim([0, 20])
plot_cutoffs(ax, mle_cutoffs, 'gray')
ax.legend(frameon=False, loc='lower right', handlelength=3)
#ax.get_legend().remove()

ax = axes_dict['c']
ax.annotate("B.", (-0.1, 0.95), xycoords="axes fraction", fontsize=28)

sns.kdeplot(df_traj_full.sort_values(by='pubs_adj', ascending=False).drop_duplicates('dblp_id').CareerAgeZero, label='Empirical', ax=ax, lw=4)
sns.kdeplot(np.argmax(sim_trajs_all, axis=0), label='Simulated', ax=ax, linestyle='--', lw=4)

ax.set_xlabel("Year $t$ of greatest productivity")
ax.set_ylabel("Density")
ax.yaxis.set_ticklabels([])
ax.set_xlim([0, 20])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.legend(frameon=False, ncol=2, loc='lower center')

ks_test_peak_year = stats.ks_2samp(
    df_traj_full[df_traj_full.index == df_traj_full.groupby('dblp_id').pubs_adj.transform('idxmax')].CareerAgeZero,
    [traj.argmax() for traj in sim_trajs_all.T]
)

ax.text(11, 0.065, f"KS$={ks_test_peak_year.statistic:.2f}$ (${print_pval(ks_test_peak_year.pvalue)}$)", fontsize=16)

sns.despine()
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig("output/model_fit.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
