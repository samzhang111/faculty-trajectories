.PHONY: clean

all: output/model_recovers_simulations.pdf output/model_fit.pdf output/model_unfit.pdf output/canonical_trajectory_and_stages.pdf

clean:
	rm output/*

output/model_fit.pdf:input/import/all-trajectories.csv input/import/full-trajectories.csv input/fit/bootstrap-all-trajectories.pickle input/simulate/simulated-all.pickle src/generate_figure_model_fit.py
	python src/generate_figure_model_fit.py

output/model_unfit.pdf:input/import/all-trajectories.csv input/import/full-trajectories.csv input/simulate/simulated-all.pickle src/generate_figure_model_unfit.py
	python src/generate_figure_model_unfit.py

output/canonical_trajectory_and_stages.pdf:input/import/all-trajectories.csv input/fit/bootstrap-all-trajectories.pickle src/generate_figure_career_stages.py
	python src/generate_figure_career_stages.py

output/model_recovers_simulations.pdf: input/fit/results-simulated-bend.pickle input/fit/results-simulated-swoop.pickle input/fit/results-simulated-realistic.pickle src/generate_figure_recover_simulated.py
	python src/generate_figure_recover_simulated.py

