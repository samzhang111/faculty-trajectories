# Stochastic dynamics of faculty productivity

This is code to accompany the manuscript "Stochastic dynamics of faculty productivity" (forthcoming), joint work with Nicholas LaBerge, Daniel B. Larremore, and Aaron Clauset.

This repository is structured in line with the [principled data processing](https://hrdag.org/2016/06/14/the-task-is-a-quantum-of-workflow/) workflow, so that it is broken down into decomposable and reproducible tasks. When a task depends on the output from a previous task, we use symlinks to join them.

For documentation purposes, we've included the directory `note`, which contains the original notebook that these ideas were tried out in.

## Initialization

1. Install the python requirements (`pip install -r requirements.txt`).
2. Run `make scaffold` to scaffold the imports and outputs of the repository.

## Running

Run `make` in the root repository. To run any specific task, change into that directory and run `make`.
