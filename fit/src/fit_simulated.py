import json
from pathlib import Path

from fit_data import fit_from_filename

with open("hand/synthetic.json") as f:
    configs = json.load(f)

for traj_config in configs["trajectories"]:
    fn = f'input/simulated-{traj_config["name"]}.csv'
    fit_from_filename(fn)
