import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy import stats
from sklearn.neighbors import KernelDensity

# A superset of the methods used in the analysis:
# some of these may be redundant.


def find_mode_kde(vals, add_noise=True):
    grid = np.linspace(vals.min(), vals.max(), 300).reshape(-1 ,1)

    if add_noise:
        vals = vals + np.random.normal(scale=1, size=len(vals))
    vals = vals.reshape(-1, 1)

    k = KernelDensity(kernel='exponential').fit(vals)
    log_dens = k.score_samples(grid)

    return grid[np.argmax(log_dens)][0]


def find_mode(vals, bins=50):
    counts, bars = np.histogram(vals, bins=bins)
    i = np.argmax(counts)
    mode = (bars[i] + bars[i+1])/2

    return mode


def trunclaplace_negloglike_single(x, alpha, mode):
    return trunclaplace_negloglike(alpha, [x], [mode])


def trunclaplace_negloglike(alpha, xs, ms):
    xs = np.array(xs)
    ms = np.array(ms)

    a = np.abs(xs - ms).sum()
    b = np.log(2 - np.exp(-ms/alpha)).sum()

    return len(xs)*np.log(alpha) + a/alpha + b


def fit_trunc_laplace(data, ms):
    result = minimize_scalar(trunclaplace_negloglike, args=(data, ms), bounds=[0, 500], method='Bounded')
    return result.x


def flatten_contingency_table(c):
    """Input: counter dict `c`, with values representing the counts of the things in keys"""

    output = []
    for key, val in c.items():
        for _ in range(int(val)):
            output.append(key)

    return output

def dist_l2(beta, x, y):
    return np.sum((beta*x - y)**2)


def dist_l1(beta, x, y):
    return np.sum(np.abs(y - beta*x))



def negloglik_fixed_intercept(beta, scale, xs, ys):
    ks = beta * xs

    if any(ks < 0):
        return np.inf

    term = 2 - np.exp(-ks/scale)

    a = np.log(term).sum()
    b = np.abs(ys - ks).sum() / scale

    return (a + b)


def mode_regression(xs, ys, prior_mode=None):
    if prior_mode is None:
        mode = find_mode(xs + ys)
        ms = np.repeat(mode, len(ys))
    else:
        ms = prior_mode

    alpha = fit_trunc_laplace(xs + ys, ms)
    result = minimize_scalar(negloglik_fixed_intercept, bounds=(0, 5), args=(alpha, xs, xs + ys))
    alpha = fit_trunc_laplace(xs + ys, xs * result.x)

    return alpha, result.x


def negloglik_l2_regression(beta, xs, ys):
    yhat = beta*xs
    residuals = ys - yhat
    alpha = stats.norm.fit(residuals, floc=0)[1]
    return -sum([stats.norm.logpdf(res, loc=0, scale=alpha) for res in residuals])


def negloglik_l1_regression(beta, xs, ys):
    yhat = beta*xs
    residuals = ys - yhat
    alpha = stats.laplace.fit(residuals, floc=0)[1]
    rv = stats.laplace(loc=0, scale=alpha)

    return -sum([rv.logpdf(res) for res in residuals])


def negloglik_mode_regression(alpha, beta, xs, ys):
    modes = xs*beta
    return trunclaplace_negloglike(alpha, xs + ys, modes)


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
        to_resample = np.where(result < trunc)[0]

    return result


def sample_trunc_laplace(loc, scale, trunc):
    """loc is a scalar or a numpy array
    trunc is the lower truncation value
    """

    rv = stats.laplace(scale=scale, loc=loc)
    result = rv.rvs()

    while result < trunc:
        result = stats.laplace(scale=scale, loc=loc).rvs()

    return result


def simulate_trajectories_using_fixed_mode(params, alpha_q0, global_mode, n=10000, Y=20):
    """
    draw q0 from exp(alpha_q0)

    the draws are from (career_start, career_end], since the data is generated from pairs of the form (q_t, q_{t+1}),
    where t ranges from [career_start, career_end]

    the distribution is TruncLaplace(q_t + global_mode, alpha_q0)
    """
    trajectories = []

    current_params = params[0]
    q0 = stats.expon.rvs(scale=alpha_q0, size=n)
    trajectories.append(q0)
    q_last = q0

    career_stage = 0
    current_params = params[career_stage]
    career_stages = sorted([x['cutoff_start'] for x in params])[1:] + [np.inf,]

    for year in range(Y):
        next_cutoff = first_above_k(career_stages, year)
        if next_cutoff > career_stages[career_stage]:
            career_stage += 1
            current_params = params[career_stage]

        mode = np.maximum(q_last + global_mode, 0)

        q_next = sample_trunc_laplace_resample_list(mode, current_params['alpha'], 0)
        trajectories.append(q_next)
        q_last = q_next

    return np.array(trajectories)


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

        q_next = sample_trunc_laplace_resample_list(mode, current_params['alpha'], 0)
        trajectories.append(q_next)
        q_last = q_next

    return np.array(trajectories)




def last_below_k(arr, k):
    """Assumes arr is sorted already, all values nonnegative"""
    last = 0
    for a in arr:
        if k < a:
            return last

        last = a

    return last

def first_above_k(arr, k, biggest=20):
    """Assumes arr is sorted already, returns `biggest` if k bigger than everything in array"""
    for a in arr:
        if k < a:
            return a

    return biggest


def aggregate_data_by_career_age(data):
    """
    `data` must be a pandas dataframe with columns `q_adj_delta` and `pubs_adj`
    """

    career_age_to_data = {}

    for year in range(0, 21):
        subset = data[
                (data.CareerAgeZero == year)
            ].sort_values(by='pubs_adj')
        career_age_to_data[year] = np.array([subset.pubs_adj, subset.q_adj_delta])

    return career_age_to_data



def last_below_k(arr, k):
    """Assumes arr is sorted already, all values nonnegative"""
    last = 0
    for a in arr:
        if k < a:
            return last

        last = a

    return last

def first_above_k(arr, k, biggest=20):
    """Assumes arr is sorted already, returns `biggest` if k bigger than everything in array"""
    for a in arr:
        if k < a:
            return a

    return biggest


def aggregate_within_cutoff(cutoffs, year, career_age_to_data):
    cutoff_start = last_below_k(cutoffs, year)
    cutoff_end = first_above_k(cutoffs, year)
    data = []
    for y in range(cutoff_start, cutoff_end):
        d = career_age_to_data[y]
        data.append(d)

    data = np.concatenate(data, axis=1)

    return data


def score_cutoffs(data, cutoff_set):
    """
    data is a pandas dataframe with `numpubs_adj` and `q_adj_delta`

    cutoff_set is an iterable of cutoffs, where a cutoff is an iterable with years of cutoff points, eg,

    cutoff_set = [(1, 2, 3), (10,), (1, 5, 10)]
    """

    regression_for_cutoffs = {}
    regression_scores = []

    alpha, global_mode = mode_regression(data.pubs_adj.values, data.q_adj_delta.values)

    q0_x = data[data.CareerAgeZero == 0].pubs_adj.values

    alpha_q0 = stats.expon.fit(q0_x)[1]
    nll_q0 = -stats.expon.logpdf(q0_x, scale=alpha_q0).sum()

    career_age_to_data = aggregate_data_by_career_age(data)
    for cutoffs in cutoff_set:
        last_cutoff = 0
        #total_nll_l1 = 0
        #total_nll_l2 = 0
        total_nll_mode = nll_q0
        total_nll_mode_fixed = nll_q0
        cutoff_data = []
        cutoff_n = []
        for cutoff in cutoffs + (np.inf,):
            data = aggregate_within_cutoff(cutoffs, last_cutoff, career_age_to_data)
            x = data[0]
            y = data[1]

            alpha, mode_beta = mode_regression(x, y, prior_mode=global_mode*x)

            mode_mu = find_mode(y)

            nll_mode = negloglik_mode_regression(alpha, mode_beta, x, y)
            nll_mode_fixed = negloglik_mode_regression(alpha, global_mode, x, y)

            #total_nll_l1 += nll_l1
            #total_nll_l2 += nll_l2
            total_nll_mode += nll_mode
            total_nll_mode_fixed += nll_mode_fixed

            cutoff_data.append(dict(
                cutoffs = cutoffs,
                cutoff_start = last_cutoff,
                cutoff_end = cutoff,
                alpha = alpha,
                mode_beta = mode_beta,
                mode_mu = mode_mu,
                nll_mode = nll_mode,
                nll_mode_fixed = nll_mode_fixed,
                n = len(x)
            ))
            cutoff_n.append(len(x))
            last_cutoff = cutoff

        regression_for_cutoffs[cutoffs] = cutoff_data
        regression_scores.append(dict(
            cutoffs=cutoffs,
            nll_mode=total_nll_mode,
            nll_mode_fixed=total_nll_mode_fixed,
            n = cutoff_n,
            min_n = min(cutoff_n)
        ))

    df_reg_scores = pd.DataFrame(regression_scores)
    df_reg_scores['num_cutoffs'] = df_reg_scores['cutoffs'].str.len()

    # We have one parameter for the initial year exponential distribution
    # For each career stage (which is changepoints + 1) we have two parameters
    # The change points themselves are also parameters
    df_reg_scores['k_varying'] = 1 + (df_reg_scores['num_cutoffs'] + 1) * 2 + df_reg_scores['num_cutoffs']
    df_reg_scores['k_fixed'] = 1 + (df_reg_scores['num_cutoffs'] + 1) + + df_reg_scores['num_cutoffs']

    df_reg_scores['aic_varying'] = df_reg_scores['nll_mode'] + df_reg_scores['k_varying']
    df_reg_scores['aic_fixed'] = df_reg_scores['nll_mode_fixed'] + df_reg_scores['k_fixed']

    df_reg_scores['bic_varying'] = df_reg_scores['nll_mode'] + df_reg_scores['k_varying']/2 * np.log(len(data))
    df_reg_scores['bic_fixed'] = df_reg_scores['nll_mode_fixed'] + df_reg_scores['k_fixed']/2 * np.log(len(data))

    return regression_for_cutoffs, df_reg_scores, global_mode, alpha_q0


def get_all_cutoffs():
    all_cutoffs = []

    for i in range(1, 20):
        all_cutoffs.append((i, ))
        for j in range(i + 1, 20):
            all_cutoffs.append((i, j))
            for k in range(j + 1, 20):
                all_cutoffs.append((i, j, k))

    return all_cutoffs


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
