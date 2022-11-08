import sys
import pickle
from pathlib import Path

import pandas as pd

from lib_fit_models import score_cutoffs, get_all_cutoffs

def fit_from_filename(fn):
    fn = Path(fn)
    df = pd.read_csv(fn)

    all_cutoffs = get_all_cutoffs()

    regression_for_cutoffs, df_reg_scores, global_mode, alpha_q0 = \
        score_cutoffs(df, all_cutoffs)

    with open(Path("output", "results-" + fn.stem + ".pickle"), "wb") as out:
        pickle.dump({
            "regression_for_cutoffs": regression_for_cutoffs,
            "df_reg_scores": df_reg_scores,
            "global_mode": global_mode,
            "alpha_q0": alpha_q0,
        }, out)


if __name__ == '__main__':
    fn = sys.argv[1]
    fit_from_filename(fn)
