import json

from simulate import simulate_trajectories_using_mode_regression,\
    trajectories_to_dataframe

with open("hand/synthetic.json") as f:
    configs = json.load(f)

N = int(configs['N'])

for trajectory_config in configs['trajectories']:
    name = trajectory_config['name']
    alpha_q0 = trajectory_config['alpha_q0']
    params = trajectory_config['parameters']

    traj_sim = simulate_trajectories_using_mode_regression(params, alpha_q0, N)
    df_traj_sim = trajectories_to_dataframe(traj_sim)

    df_traj_sim.to_csv(f"output/simulated-{name}.csv", index=False)
