import argparse
import pickle

from simulate import simulate_trajectories_using_mode_regression,\
    trajectories_to_dataframe


parser = argparse.ArgumentParser(
                    prog = 'GenerateTrajectories',
                    description = 'Generate trajectories from fitted params',
                    epilog = "That's all, folks")

parser.add_argument('pickle', help='path to pickled results of the fit')
parser.add_argument('outpickle', help='filename to save pickled output')
parser.add_argument('-N', '--trajectories', type=int,
                    help='number of trajectories to generate')

args = parser.parse_args()
with open(args.pickle, "rb") as f:
    results = pickle.load(f)

N = args.trajectories

mle_cutoffs = results['df_reg_scores'].sort_values(by='aic_varying').cutoffs.iloc[0]
trajs = simulate_trajectories_using_mode_regression(results['regression_for_cutoffs'][mle_cutoffs], results['alpha_q0'], N)
df_trajs = trajectories_to_dataframe(trajs)

with open(args.outpickle, "wb") as out:
    pickle.dump({
        'trajs': trajs,
        'df_trajs': df_trajs,
    }, out)
