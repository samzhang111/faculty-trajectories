import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lib_fit_models import score_cutoffs, get_all_cutoffs


def block_bootstrap(dataframe, group_id='dblp_id'):
    """
    Bootstrapping data within blocks (such as individuals).
    """

    group_ids = dataframe[group_id].unique()
    shuf_group_ids = np.random.choice(group_ids, size=len(group_ids),
                                      replace=True)

    return pd.merge(
        dataframe,
        pd.Series(shuf_group_ids).reset_index().rename(columns={0: 'dblp_id'}),
        on='dblp_id'
    ).drop(['index'], axis=1)

n_bootstrap = int(sys.argv[1])

fn_df = Path(sys.argv[2])
df = pd.read_csv(fn_df)

fn_results = Path(sys.argv[3])
with open(fn_results, "rb") as p:
    results = pickle.load(p)

top_cutoffs_50 = results["df_reg_scores"].sort_values(by="aic_varying").\
    cutoffs.head(50)

regression_for_cutoffs_bootstrap = []
df_reg_scores_bootstrap = []
global_mode_bootstrap = []
alpha_q0_bootstrap = []

for i in range(n_bootstrap):
    if i % 10 == 0:
        print('.', end='')
    df_boot = block_bootstrap(df)
    rfc, dfr, gm, aq0 = score_cutoffs(df_boot, top_cutoffs_50)
    regression_for_cutoffs_bootstrap.append(rfc)
    df_reg_scores_bootstrap.append(dfr.sort_values(by='aic_varying'))
    global_mode_bootstrap.append(gm)
    alpha_q0_bootstrap.append(aq0)

with open(Path("output", "bootstrap-" + fn_df.stem + ".pickle"), "wb") as out:
    pickle.dump({
        'regression_for_cutoffs_bootstrap': regression_for_cutoffs_bootstrap,
        'df_reg_scores_bootstrap': df_reg_scores_bootstrap,
        'global_mode_bootstrap': global_mode_bootstrap,
        'alpha_q0_bootstrap': alpha_q0_bootstrap,
    }, out)
