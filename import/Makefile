.PHONY: clean

all: output/all-trajectories.csv output/simulated.touch

clean:
	rm output/*

output/all-trajectories.csv:input src/import_trajectories.py
	python src/import_trajectories.py

output/simulated.touch:hand/synthetic.json
	python src/generate_synthetic_trajectories.py
	touch output/simulated.touch
