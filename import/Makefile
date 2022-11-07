.PHONY: clean

all: output/all-trajectories.csv

clean:
	rm output/*

output/all-trajectories.csv:input
	python src/import_trajectories.py
