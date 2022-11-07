.PHONY: import fit simulate write scaffold clean

all: import fit simulate write

import:
	cd import && make

fit:
	cd fit && make

simulate:
	cd simulate && make

write:
	cd write && make

scaffold:
	mkdir -p import/output fit/output simulate/output write/output write/input
	-cd fit && ln -s ../import/output input 
	-cd simulate && ln -s ../fit/output input
	-cd write/input && ln -s ../../simulate/output simulate
	-cd write/input && ln -s ../../fit/output fit
	-cd write/input && ln -s ../../import/output import
