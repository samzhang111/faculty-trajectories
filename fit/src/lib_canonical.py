import statsmodels.api as sm
import numpy as np


def aicc(mod_result):
    return sm.tools.eval_measures.aicc(mod_result.llf, mod_result.nobs, mod_result.df_model)


def fit_linear(traj):
    model_linear = sm.OLS(
        traj,
        sm.add_constant(range(len(traj))),
    )

    results_linear = model_linear.fit()
    return aicc(results_linear)


def fit_piecewise_linear(traj, t, verbose=False):
    traj_before_t = traj[:t]
    traj_after_t = traj[t:]

    model_breakpoint_before_t = sm.OLS(
        traj_before_t,
        sm.add_constant(range(len(traj_before_t))),
    )

    results_breakpoint_before_t = model_breakpoint_before_t.fit()

    model_breakpoint_after_t = sm.OLS(
        traj_after_t,
        sm.add_constant(range(len(traj_after_t))),
    )

    results_breakpoint_after_t = model_breakpoint_after_t.fit()

    #breakpoint_aic = -2*(results_breakpoint_after_t.llf + results_breakpoint_before_t.llf) + 8
    breakpoint_aicc = sm.tools.eval_measures.aicc(results_breakpoint_after_t.llf + results_breakpoint_before_t.llf, len(traj), 4)
    if verbose:
        print(f"Before SSE:{results_breakpoint_before_t.mse_total}*{results_breakpoint_before_t.nobs}, After MSE: {results_breakpoint_after_t.mse_total}*{results_breakpoint_after_t.nobs}")

    return breakpoint_aicc, results_breakpoint_before_t.params[1], results_breakpoint_after_t.params[1]


def find_breakpoint(traj):
    traj = traj[:20]
    linear_aicc = fit_linear(traj)
    best_aicc = linear_aicc
    best_params = [np.nan, np.nan]
    best_tstar = -1

    for t in range(3, len(traj) - 3):
        current_aicc, m1, m2 = fit_piecewise_linear(traj, t)
        if current_aicc < best_aicc:
            best_aicc = current_aicc
            best_params = (m1, m2)
            best_tstar = t

    return best_tstar, best_params[0], best_params[1]
