.PHONY: clean

all: output/results-full-trajectories.pickle output/results-all-trajectories.pickle \
	output/bootstrap-all-trajectories.pickle output/bootstrap-full-trajectories.pickle

clean:
	rm output/*

output/results-all-trajectories.pickle:input/all-trajectories.csv
	python src/fit_data.py $<

output/results-full-trajectories.pickle:input/full-trajectories.csv
	python src/fit_data.py $<

output/bootstrap-full-trajectories.pickle:input/full-trajectories.csv output/results-full-trajectories.pickle
	python src/fit_bootstrap.py 1000 $?

output/bootstrap-all-trajectories.pickle:input/all-trajectories.csv output/results-all-trajectories.pickle
	python src/fit_bootstrap.py 1000 $?
