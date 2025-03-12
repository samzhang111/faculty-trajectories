import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator, PercentFormatter
import seaborn as sns
from scipy import stats

from plot_helpers import *

init_plot()

sns.color_palette(colors + lighter_colors)


def find_mode(vals, bins=100):
    counts, bars = np.histogram(vals, bins=bins)
    i = np.argmax(counts)
    mode = (bars[i] + bars[i+1])/2

    return mode

#######################################
# Load data
#######################################

with open("input/simulate/simulated-all.pickle", "rb") as f:
    sim_results = pickle.load(f)
    trajs = sim_results['trajs']
    df_trajs_sim = sim_results['df_trajs']

df_traj_all = pd.read_csv("input/import/all-trajectories.csv")
df_traj_full = pd.read_csv("input/import/full-trajectories.csv")

#######################################
# Create auxiliary data structures
#######################################

empirical_zero_run_lengths = list(itertools.chain(*df_traj_all.groupby('dblp_id').pubs_adj.apply(count_zero_runs)))
hazard_counts_emp, hazard_emp, hazard_emp_vars = hazard_from_runlengths(empirical_zero_run_lengths)

empirical_zero_run_lengths_full = list(itertools.chain(*df_traj_full.groupby('dblp_id').pubs_adj.apply(count_zero_runs)))
hazard_counts_emp_full, hazard_emp_full, hazard_emp_vars_full = hazard_from_runlengths(empirical_zero_run_lengths_full)

#######################################
# Generate plot
#######################################

hatch = '/'
ks_test = stats.ks_2samp(
    df_traj_full.groupby('dblp_id').pubs_adj.std(),
    trajs.std(axis=0)
)


fig, axes = plt.subplots(2, 2, figsize=(15, 15), dpi=200)

zero_run_lengths = list(itertools.chain(*[count_zero_runs(t) for t in np.floor(trajs.T)]))
hazard_counts_sim, hazard_sim, hazard_sim_vars = hazard_from_runlengths(zero_run_lengths)

df_year_5 = pd.DataFrame([
    df_traj_all[df_traj_all.CareerAge == 5].cumpubs.values,
    np.cumsum(trajs, axis=0)[5,:df_traj_all[df_traj_all.CareerAge == 5].shape[0]]
]).T

df_year_5.columns = ['Empirical', 'Simulated']

ax = axes[0][0]
ax.annotate("A.", (-0.2, 0.95), xycoords="axes fraction", fontsize=24)

lw = 3
emp = df_traj_full.groupby('dblp_id').pubs_adj.std()
sim = df_trajs_sim.groupby('ix').pubs_adj.std()
sns.kdeplot(emp, label=None, ax=ax, color=colors[0], lw=lw)
sns.kdeplot(df_traj_full[np.floor(df_traj_full.pubs_adj) > 0].groupby('dblp_id').pubs_adj.std(), label='Empirical (no zeros)', ax=ax, color=colors[0], linestyle=':', lw=lw)
sns.kdeplot(sim, label=None, ax=ax, color=colors[1], lw=lw)
sns.kdeplot(df_trajs_sim[np.floor(df_traj_all.pubs_adj) > 0].groupby('ix').pubs_adj.std(), label='Simulated (no zeros)', ax=ax, color=colors[1], linestyle=":", lw=lw)
ax.set_xlim([0, 15])

#emp_mode = find_mode(emp.values)
#sim_mode = find_mode(sim.values)

ax.set_xlabel("Career std. dev. of annual productivity, $\sigma$")
ax.yaxis.set_ticklabels([])
ax.set_ylabel("Density\n\n")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.text(8, 0.12, f"Empirical vs. simulated:\nKS$={ks_test.statistic:.2f}$ (${print_pval(ks_test.pvalue)}$)", fontsize=14)


#ax.axvline(emp_mode, color=colors[0], lw=2)
#ax.axvline(sim_mode, color=colors[1], lw=2)
#ax.scatter(emp_mode, 0, marker='x', s=100, color=colors[0], zorder=10, clip_on=False)
#ax.scatter(sim_mode, 0, marker='o', s=100, color=colors[1], zorder=10, clip_on=False)

emp_legend = mlines.Line2D([], [], color=colors[0],# marker='x',
                          markersize=10, label='Empirical')

sim_legend = mlines.Line2D([], [], color=colors[1],# marker='o',
                          markersize=10, label='Simulated')

emp_legend_no_zeros = mlines.Line2D([], [], color=colors[0], linestyle=':',
                          label='Empirical (no zeros)')

sim_legend_no_zeros = mlines.Line2D([], [], color=colors[1], linestyle=':',
                          label='Simulated (no zeros)')

ax.legend(handles=[emp_legend, emp_legend_no_zeros,
                   sim_legend, sim_legend_no_zeros],
          frameon=False, loc='upper right')
#ax.legend(frameon=False, loc='upper right')

ax = axes[0][1]
ax.annotate("B.", (-0.2, 0.95), xycoords="axes fraction", fontsize=24)

rounded_sim = np.floor(trajs[np.where(trajs <= 20)])
p_sim = sum(rounded_sim.flatten() == 0) / len(rounded_sim.flatten())
n_sim = len(rounded_sim.flatten())
ci_sim = wald_interval_95(p_sim, n_sim)

rounded_emp = np.floor(df_traj_full[df_traj_full.pubs_adj <= 20].pubs_adj)
p_emp = sum(rounded_emp == 0) / len(rounded_emp)
n_emp = len(rounded_emp)
ci_emp = wald_interval_95(p_emp, n_emp)

sns.histplot(rounded_emp, label='Empirical', stat="probability", color=lighter_colors[0], binwidth=1, alpha=0.8, linewidth=1.5, ax=ax)
sns.histplot(rounded_sim, label='Simulated', stat="probability", color=lighter_colors[1], binwidth=1, alpha=0.6, linewidth=1.5, ax=ax, hatch=hatch)

print(f"P(zero) in simulated: {p_sim:.2f}\nP(zero) in emp: {p_emp:.2f}")

ax.errorbar(x=0.5, y=p_emp, yerr=ci_emp[1] - p_emp, color="black", capsize=4, elinewidth=3, capthick=2)
ax.errorbar(x=0.5, y=p_sim, yerr=ci_sim[1] - p_sim, color="black", capsize=4, elinewidth=3, capthick=2)

ax.legend(frameon=False)
ax.set_xlabel("Productivity")
ax.set_ylabel("Percent")
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))




ax = axes[1][0]
ax.annotate("C.", (-0.2, 0.95), xycoords="axes fraction", fontsize=24)
ttest_year_5 = stats.ttest_ind(df_year_5.Empirical, df_year_5.Simulated, equal_var=False)
yname = 'Cumulative publications at $t=5$'
sns.violinplot(x='x', y=yname, data=df_year_5.melt(value_name=yname, var_name='x'), ax=ax, saturation=0.8, palette=lighter_colors)
ax.set_xlabel(None)
ax.text(0.6, 260, f"Empirical vs. simulated:\n$t={ttest_year_5.statistic:.2f}$ (${print_pval(ttest_year_5.pvalue)}$)", fontsize=14)
ax.yaxis.grid(linewidth=1)




ax = axes[1][1]
ax.annotate("D.", (-0.2, 0.95), xycoords="axes fraction", fontsize=24)
# Compute empirical survival function
empirical_survival = []
total_count_emp = sum(hazard_counts_emp_full[i]['N'] + hazard_counts_emp_full[i]['Y'] for i in range(1, 15))
cumulative_count_emp = 0

for i in range(1, 15):
    cumulative_count_emp += hazard_counts_emp_full[i]['N'] + hazard_counts_emp_full[i]['Y']
    empirical_survival.append(1 - cumulative_count_emp / total_count_emp)

# Compute simulated survival function
simulated_survival = []
total_count_sim = sum(hazard_counts_sim[i]['N'] + hazard_counts_sim[i]['Y'] for i in range(1, 15))
cumulative_count_sim = 0

for i in range(1, 15):
    cumulative_count_sim += hazard_counts_sim[i]['N'] + hazard_counts_sim[i]['Y']
    simulated_survival.append(1 - cumulative_count_sim / total_count_sim)

# Plot survival functions
ax.plot(range(1, 15), empirical_survival, label='Empirical', color=lighter_colors[0], linewidth=2)
ax.plot(range(1, 15), simulated_survival, label='Simulated', linestyle='dashed', color=lighter_colors[1], linewidth=2)

ax.set_yscale("log")
ax.set_xlabel("Zeros per trajectory")
ax.set_ylabel("Survival Probability")
ax.legend(frameon=False)
ax.annotate(f"$N={df_traj_full.dblp.nunique()}$ trajectories each", (0.5, 0.7), xycoords="axes fraction", fontsize=14)

sns.despine()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
sns.despine()

plt.savefig("output/model_unfit.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
