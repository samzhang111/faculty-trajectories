.PHONY: clean

all: output/model_fit.pdf output/model_unfit.pdf

clean:
	rm output/*

output/model_fit.pdf:input/import/all-trajectories.csv input/import/full-trajectories.csv input/fit/bootstrap-all-trajectories.pickle input/simulate/simulated-all.pickle
	python src/generate_figure_model_fit.py

output/model_unfit.pdf:input/import/all-trajectories.csv input/import/full-trajectories.csv input/simulate/simulated-all.pickle
	python src/generate_figure_model_unfit.py
