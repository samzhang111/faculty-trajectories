import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


from plot_helpers import *

init_plot()

#######################################
# Import data
#######################################

N_emp = 10000

df_traj_sim_bend = pd.read_csv("input/import/simulated-bend.csv")
df_traj_sim_swoop = pd.read_csv("input/import/simulated-swoop.csv")
df_traj_sim_realistic = pd.read_csv("input/import/simulated-realistic.csv")

with open("input/simulate/simulated-bend.pickle", "rb") as f:
    pickle_bend = pickle.load(f)
    traj_sim_bend = pickle_bend['trajs']
    cutoffs_sim_bend = pickle_bend['cutoffs']


with open("input/simulate/simulated-swoop.pickle", "rb") as f:
    pickle_swoop = pickle.load(f)
    traj_sim_swoop = pickle_swoop['trajs']
    cutoffs_sim_swoop = pickle_swoop['cutoffs']

with open("input/simulate/simulated-realistic.pickle", "rb") as f:
    pickle_realistic = pickle.load(f)
    traj_sim_realistic = pickle_realistic['trajs']
    cutoffs_sim_realistic = pickle_realistic['cutoffs']

#######################################
# Make plot
#######################################

enable_matplotlib_latex()

fig, axes = plt.subplots(3, 1, figsize=(8, 15), dpi=200)

ax = axes[0]
ax.annotate("A.", (-0.15, 0.97), xycoords="axes fraction", fontsize=24)

sns.lineplot(x=df_traj_sim_swoop.head(N_emp).CareerAgeZero.values, y=df_traj_sim_swoop.head(N_emp).pubs_adj.values, ax=ax, label='Original', estimator=np.mean, lw=4, **marker_settings(colors[0], markersize=10))
plot_trajectories(ax, traj_sim_swoop, label='Recovered', markersize=10)
plot_cutoffs(ax, cutoffs_sim_swoop, 'gray')

ax.set_xlabel(None)
ax.set_ylabel("$q_t$")

ax.set_xlim([0, 19])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(loc='upper left', frameon=False, handlelength=3)



ax = axes[1]
ax.annotate("B.", (-0.15, 0.97), xycoords="axes fraction", fontsize=24)

sns.lineplot(x=df_traj_sim_bend.head(N_emp).CareerAgeZero.values, y=df_traj_sim_bend.head(N_emp).pubs_adj.values, ax=ax, label='Original', estimator=np.mean, **marker_settings(colors[0]))
plot_trajectories(ax, traj_sim_bend, label='Recovered')
plot_cutoffs(ax, cutoffs_sim_bend, 'gray')

ax.set_xlabel(None)
ax.set_ylabel("$q_t$")

ax.set_xlim([0, 19])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.legend(loc='lower right', frameon=False, handlelength=1)


ax.get_legend().remove()

ax = axes[2]
ax.annotate("C.", (-0.15, 0.97), xycoords="axes fraction", fontsize=24)

sns.lineplot(x=df_traj_sim_realistic.head(N_emp).CareerAgeZero.values, y=df_traj_sim_realistic.head(N_emp).pubs_adj.values, ax=ax, label='Original', estimator=np.mean, **marker_settings(colors[0]))
plot_trajectories(ax, traj_sim_realistic, label='Recovered')
plot_cutoffs(ax, cutoffs_sim_realistic, 'gray')

ax.set_xlabel("Year, $t$")
ax.set_ylabel("$q_t$")

ax.set_xlim([0, 19])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.legend(loc='lower right', frameon=False, handlelength=1)

ax.get_legend().remove()

sns.despine()

plt.savefig("./output/model_recovers_simulations.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
