import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from curlyBrace import curlyBrace


from plot_helpers import *

init_plot()

#######################################
# Import data
#######################################

# Empirical trajectories
df_traj_all = pd.read_csv("input/import/all-trajectories.csv")

# Bootstrapped fits
with open("input/fit/bootstrap-all-trajectories.pickle", "rb") as f:
    bootstrap_results = pickle.load(f)

alpha_q0_bootstrap = bootstrap_results['alpha_q0_bootstrap']

with open("input/simulate/simulated-all.pickle", "rb") as f:
    simulated_results_all = pickle.load(f)
    sim_trajs_all = simulated_results_all['trajs']

#######################################
# Make canonical trajectory + career stage plot
#######################################

enable_matplotlib_latex()

fig, axes_dict = plt.subplot_mosaic("""
aabbccdd
aabbccdd
eeeeeeee
eeeeeeee
eeeeeeee
""", figsize=(16, 10), dpi=200)

bbox_to_anchor = (0, 0.95, 1, 0.1)
yrs_b = df_traj_all[(df_traj_all.CareerAgeZero > 0) & (df_traj_all.CareerAgeZero <= 4)]
yrs_c = df_traj_all[(df_traj_all.CareerAgeZero > 4) & (df_traj_all.CareerAgeZero <= 7)]
yrs_d = df_traj_all[(df_traj_all.CareerAgeZero > 7)]

fit_label = "\\text{{Fit:~}}"
handlelength = 1

ax = axes_dict['a']
ax.annotate("A.", (-0.4, 0.95), xycoords="axes fraction", fontsize=24)

first_years = df_traj_all[df_traj_all.CareerAgeZero == 0]

q0_lambda = first_years.pubs_adj.mean()
q0_rv = stats.expon(loc=0, scale=q0_lambda)
x = np.linspace(0, 25, 100)

sns.histplot(first_years.pubs_adj, ax=ax, stat='density', label="All (empirical)", alpha=0.2, lw=0.5, edgecolor=colors[0])
ax.plot(x, q0_rv.pdf(x), lw=2, label=f'Fit: $\widehat{{\lambda}} = {q0_lambda:.2f}$', color='black', linestyle='--')
ax.set_xlim([0, 20])
ax.set_ylim([1e-3*5, .3])
ax.set_ylabel("Density")
ax.semilogy()
handles, labels = ax.get_legend_handles_labels()
handles.pop(1) # Duplicate legend label for some reason. Drop it.
labels.pop(1)
ax.legend(handles[::-1], labels[::-1], loc='upper center', frameon=False, bbox_to_anchor=(0, 1, 1, 0.1), handlelength=handlelength)
ax.set_xlabel("First year productivity, $q_0$")


axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(.7, .55, .4, .25),
                   bbox_transform=ax.transAxes)
sns.kdeplot(alpha_q0_bootstrap, fill=True, ax=axins, label=None)
axins.tick_params(left=True, right=False, labelleft=False, labelright=False)
axins.set_xlabel("$\widehat{\lambda}$ bootstraps")
axins.xaxis.label.set_size(12)
axins.tick_params(axis='x', labelsize=12)
axins.set_ylabel(None)
axins.axvline(q0_lambda, linestyle='--', lw=1, color='black')
axins.xaxis.set_major_locator(MaxNLocator(4))



ax = axes_dict['b']
ax.annotate("B.", (-0.4, 0.95), xycoords="axes fraction", fontsize=24)

sns.histplot(yrs_b.q_adj_delta.dropna(), ax=ax, stat='density', binrange=(-15, 15), bins=25, lw=0.5, edgecolor=colors[0], alpha=0.2)
_, mle_alpha, mle_mode = plot_laplace_fit(ax, yrs_b.q_adj_delta.dropna(),fit_label, "black", linestyle="--", lw=2)
ax.legend(loc='upper left', frameon=False, bbox_to_anchor=bbox_to_anchor, handlelength=handlelength)

ax.set_xlabel("$q_{t+1} - q_t$")
ax.set_ylabel(None)
ax.set_xlim([-15, 15])
ax.set_ylim([1e-4 * 7, 1])
ax.axvline(0, linestyle=':', color='gray', lw=1)

ax = axes_dict['c']
ax.annotate("C.", (-0.4, 0.95), xycoords="axes fraction", fontsize=24)

sns.histplot(yrs_c.q_adj_delta.dropna(), ax=ax, stat='density', binrange=(-15, 15), bins=25, lw=0.5, edgecolor=colors[0], alpha=0.2)
_, mle_alpha, mle_mode = plot_laplace_fit(ax, yrs_c.q_adj_delta.dropna(), fit_label, "black", linestyle="--", lw=2)
ax.legend(loc='upper left', frameon=False, bbox_to_anchor=bbox_to_anchor, handlelength=handlelength)
ax.set_xlabel("$q_{t+1} - q_t$")
ax.set_ylabel(None)
ax.set_xlim([-15, 15])
ax.set_ylim([1e-4 * 7, 1])
ax.axvline(0, linestyle=':', color='gray', lw=1)

ax = axes_dict['d']
ax.annotate("D.", (-0.4, 0.95), xycoords="axes fraction", fontsize=24)

sns.histplot(yrs_d.q_adj_delta.dropna(), ax=ax, stat='density', binrange=(-15, 15), bins=25, lw=0.5, edgecolor=colors[0], alpha=0.2)
_, mle_alpha, mle_mode = plot_laplace_fit(ax, yrs_d.q_adj_delta.dropna(), fit_label, "black", linestyle="--", lw=2)
ax.legend(loc='upper left', frameon=False, bbox_to_anchor=bbox_to_anchor, handlelength=handlelength)
ax.set_xlabel("$q_{t+1} - q_t$")
ax.set_ylabel(None)
ax.set_xlim([-15, 15])
ax.set_ylim([1e-4 * 7, 1])
ax.axvline(0, linestyle=':', color='gray', lw=1)


ax = axes_dict['e']
ax.annotate("E.", (-0.1, 0.95), xycoords="axes fraction", fontsize=24)

sns.lineplot(x=df_traj_all.CareerAgeZero, y=df_traj_all.pubs_adj,ax=ax, lw=4, **marker_settings(colors[0], markersize=12))
ax.set_clip_on(False)
ax.set_xlabel("Year $t$ since first year as assistant professor")
ax.set_ylabel("Mean annual productivity, $q_t$")
ax.set_xlim([0, 20])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_ticks(range(0, 21))
ax.yaxis.grid(linewidth=1)
plt.setp(ax.collections, clip_on=False)
plt.setp(ax.lines, clip_on=False, zorder=5)

font = {'color': 'black', 'size': 20}
h = 8
gap = 0.1
curlyBrace(fig, ax, (-0.1, h), (0.8, h), str_text="A.", color='black', k_r=0.3, fontdict=font)
curlyBrace(fig, ax, (0.8 + gap, h), (4.5 - gap, h), str_text="B.", color='black', k_r=0.08, fontdict=font)
curlyBrace(fig, ax, (4.5 + gap, h), (7.5 - gap, h), str_text="C.", color='black', k_r=0.11, fontdict=font)
curlyBrace(fig, ax, (7.5 + gap, h), (20, h), str_text="D.", color='black', k_r=0.027, fontdict=font)
#curlyBrace(fig, ax, (14.5 + gap, h), (20, h), str_text="D.", color='black', k_r=0.05, fontdict=font)

plt.subplots_adjust(wspace=1, hspace=1.5)

sns.despine()
plt.savefig("output/canonical_trajectory_and_stages.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
