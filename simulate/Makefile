.PHONY: clean aarc

SRC = src/generate_trajectories.py
INPUT_DIR = input
OUTPUT_DIR = output

CSV_FILES = \
    aarc_trajectories_biological_sciences \
    aarc_trajectories_chemistry \
    aarc_trajectories_computer_science \
    aarc_trajectories_economics \
    aarc_trajectories_electrical_engineering \
    aarc_trajectories_history \
    aarc_trajectories_management \
    aarc_trajectories_mathematics \
    aarc_trajectories_mechanical_engineering \
    aarc_trajectories_physics \
    aarc_trajectories_political_science \
    aarc_trajectories_psychology \
    aarc_trajectories_sociology

# Turn input/results-aarc_trajectories_X.pickle -> output/simulated-aarc_trajectories_X.pickle
AARC_SIMULATED = $(CSV_FILES:%=$(OUTPUT_DIR)/simulated-%.pickle)

all: \
	$(OUTPUT_DIR)/simulated-all.pickle \
	$(OUTPUT_DIR)/simulated-full.pickle \
	$(OUTPUT_DIR)/simulated-bend.pickle \
	$(OUTPUT_DIR)/simulated-swoop.pickle \
	$(OUTPUT_DIR)/simulated-realistic.pickle \
	$(AARC_SIMULATED)

clean:
	rm -f $(OUTPUT_DIR)/*

$(OUTPUT_DIR)/simulated-%.pickle: $(INPUT_DIR)/results-%.pickle $(SRC)
	python $(SRC) $< $@ -N 10000

aarc: $(AARC_SIMULATED)

$(OUTPUT_DIR)/simulated-all.pickle: $(INPUT_DIR)/results-all-trajectories.pickle $(SRC)
	python $(SRC) $< $@ -N 10000

$(OUTPUT_DIR)/simulated-full.pickle: $(INPUT_DIR)/results-full-trajectories.pickle $(SRC)
	python $(SRC) $< $@ -N 10000

$(OUTPUT_DIR)/simulated-bend.pickle: $(INPUT_DIR)/results-simulated-bend.pickle $(SRC)
	python $(SRC) $< $@ -N 10000

$(OUTPUT_DIR)/simulated-swoop.pickle: $(INPUT_DIR)/results-simulated-swoop.pickle $(SRC)
	python $(SRC) $< $@ -N 10000

$(OUTPUT_DIR)/simulated-realistic.pickle: $(INPUT_DIR)/results-simulated-realistic.pickle $(SRC)
	python $(SRC) $< $@ -N 10000

