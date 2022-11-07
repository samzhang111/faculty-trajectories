.PHONY: clean

all: output/simulated-all.pickle output/simulated-full.pickle

clean:
	rm output/*

output/simulated-all.pickle:input/results-all-trajectories.pickle
	python src/generate_trajectories.py $< $@ -N 10000

output/simulated-full.pickle:input/results-full-trajectories.pickle
	python src/generate_trajectories.py $< $@ -N 10000

