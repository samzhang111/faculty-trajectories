.PHONY: clean aarc

SRC = src
INPUT_DIR = input
OUTPUT_DIR = output

CSV_FILES = \
    aarc_trajectories_biological_sciences.csv \
    aarc_trajectories_chemistry.csv \
    aarc_trajectories_computer_science.csv \
    aarc_trajectories_economics.csv \
    aarc_trajectories_electrical_engineering.csv \
    aarc_trajectories_history.csv \
    aarc_trajectories_management.csv \
    aarc_trajectories_mathematics.csv \
    aarc_trajectories_mechanical_engineering.csv \
    aarc_trajectories_physics.csv \
    aarc_trajectories_political_science.csv \
    aarc_trajectories_psychology.csv \
    aarc_trajectories_sociology.csv

# Turn aarc_trajectories_X.csv -> output/results-X.pickle
AARC_FIELDS = $(CSV_FILES:%.csv=$(OUTPUT_DIR)/results-%.pickle)

all: output/results-full-trajectories.pickle output/results-all-trajectories.pickle \
	output/bootstrap-all-trajectories.pickle output/bootstrap-full-trajectories.pickle \
	output/results-simulated.touch $(AARC_FIELDS)

clean:
	rm output/*

aarc: $(AARC_FIELDS)


$(OUTPUT_DIR)/results-%.pickle: $(INPUT_DIR)/%.csv $(SRC)
	python $(SRC)/fit_data.py $<

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
