.PHONY: clean

all: output/model_fit.pdf output/model_unfit.pdf output/canonical_trajectory_and_stages.pdf

clean:
	rm output/*

output/model_fit.pdf:input/import/all-trajectories.csv input/import/full-trajectories.csv input/fit/bootstrap-all-trajectories.pickle input/simulate/simulated-all.pickle
	python src/generate_figure_model_fit.py

output/model_unfit.pdf:input/import/all-trajectories.csv input/import/full-trajectories.csv input/simulate/simulated-all.pickle
	python src/generate_figure_model_unfit.py

output/canonical_trajectory_and_stages.pdf:input/import/all-trajectories.csv input/fit/bootstrap-all-trajectories.pickle
	python src/generate_figure_career_stages.py
