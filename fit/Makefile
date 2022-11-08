.PHONY: clean

all: output/results-full-trajectories.pickle output/results-all-trajectories.pickle \
	output/bootstrap-all-trajectories.pickle output/bootstrap-full-trajectories.pickle \
	output/results-simulated.touch

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

output/results-simulated.touch:input/simulated-*.csv hand/synthetic.json
	python src/fit_simulated.py
