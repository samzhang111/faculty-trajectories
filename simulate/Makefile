.PHONY: clean

SRC=src/generate_trajectories.py


all: output/simulated-all.pickle output/simulated-full.pickle output/simulated-bend.pickle output/simulated-swoop.pickle output/simulated-realistic.pickle

clean:
	rm output/*

output/simulated-all.pickle:input/results-all-trajectories.pickle $(SRC)
	python src/generate_trajectories.py $< $@ -N 10000

output/simulated-full.pickle:input/results-full-trajectories.pickle $(SRC)
	python src/generate_trajectories.py $< $@ -N 10000

output/simulated-bend.pickle: input/results-simulated-bend.pickle $(SRC)
	python src/generate_trajectories.py $< $@ -N 10000

output/simulated-swoop.pickle: input/results-simulated-swoop.pickle $(SRC)
	python src/generate_trajectories.py $< $@ -N 10000

output/simulated-realistic.pickle: input/results-simulated-realistic.pickle $(SRC)
	python src/generate_trajectories.py $< $@ -N 10000

