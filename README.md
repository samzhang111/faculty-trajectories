# Changes in variance explain scientific productivity patterns across career stages

This is code to accompany the manuscript "Changes in variance explain scientific productivity patterns across career stages" (forthcoming), joint work with Nicholas LaBerge, Daniel B. Larremore, and Aaron Clauset.

This repository is structured in line with the [principled data processing](https://hrdag.org/2016/06/14/the-task-is-a-quantum-of-workflow/) workflow, so that it is broken down into decomposable and reproducible tasks. When a task depends on the output from a previous task, we use symlinks to join them.

For documentation purposes, we've included the directory `note`, which contains the original notebook that these ideas were tried out in.

## Initialization

1. Install the python requirements (`pip install -r requirements.txt`).
2. Run `make scaffold` to scaffold the imports and outputs of the repository.
3. Place the raw data file `adjusted_productivity.csv` into `import/input`.

## Running

Run `make` in the root repository. To run any specific task, change into that directory and run `make`.

## Data

For ease of replicability, we include the data in `import/input/adjusted_productivities.csv`. This data is directly taken from [Way, Morgan, Clauset, and Larremore (2017)](https://www.pnas.org/doi/full/10.1073/pnas.1702121114), with slightly stricter inclusion criteria (see paper).
