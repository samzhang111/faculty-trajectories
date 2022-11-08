import numpy as np
import pandas as pd
from scipy import stats


def trajectories_to_dataframe(trajs):
    dfs = []

    for i in range(trajs.shape[1]):
        df_sim = pd.DataFrame(zip(range(21), trajs[:,i]), columns=['CareerAgeZero', 'pubs_adj'])
        df_sim['ix'] = i

        dfs.append(df_sim)

    df_sim = pd.concat(dfs)
    df_sim['pubs_adj_next'] = df_sim.groupby(['ix']).pubs_adj.shift(periods=-1)
    df_sim['q_adj_delta'] = df_sim.pubs_adj_next - df_sim.pubs_adj

    return df_sim.dropna()


def sample_trunc_laplace_resample_list(loc, scale, trunc):
    """loc is a numpy array
    trunc is the lower truncation value
    """

    rv = stats.laplace(scale=scale, loc=loc)
    result = rv.rvs()
    to_resample = np.where(result < trunc)[0]

    while len(to_resample) > 0:
        try:
            resampled = stats.laplace(scale=scale, loc=loc[to_resample]).rvs()
        except TypeError:
            resampled = stats.laplace(scale=scale, loc=loc).rvs()

        result[to_resample] = resampled
        to_resample = np.where(result < 0)[0]

    return result


def first_above_k(arr, k, biggest=20):
    """Assumes arr is sorted already, returns `biggest` if k bigger than everything in array"""
    for a in arr:
        if k < a:
            return a

    return biggest


def simulate_trajectories_using_mode_regression(params, alpha_q0, n=10000, global_mode=None):
    """
    draw q0 from exp(alpha_q0)

    the draws are from (career_start, career_end], since the data is generated from pairs of the form (q_t, q_{t+1}),
    where t ranges from [career_start, career_end]

    """
    trajectories = []

    current_params = params[0]
    q0 = stats.expon.rvs(scale=alpha_q0, size=n)
    trajectories.append(q0)
    q_last = q0

    career_stage = 0
    current_params = params[career_stage]
    career_stages = sorted([x['cutoff_start'] for x in params])[1:] + [np.inf,]

    for year in range(20):
        next_cutoff = first_above_k(career_stages, year)
        if next_cutoff > career_stages[career_stage]:
            career_stage += 1
            current_params = params[career_stage]

        if global_mode:
            mode = q_last * global_mode
        else:
            mode = q_last * current_params['mode_beta']

        q_next = sample_trunc_laplace_resample_list(mode, current_params['alpha'], -q_last)
        trajectories.append(q_next)
        q_last = q_next

    return np.array(trajectories)
