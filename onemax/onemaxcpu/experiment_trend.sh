#!/bin/bash

CWD=$(pwd); readonly CWD

RUNALL=10
GenomeLength=$(seq 128 128 1024)
PopulationSize=$(seq 128 128 1024)
DATETIME=$(date +%Y%m%d-%H%M%S)
readonly FILENAME=${CWD}/onemax_prms.dat

for popsize in ${PopulationSize}; do
	sed -i "s/^POP_SIZE.*$/POP_SIZE ${popsize}/" "${FILENAME}"
	for genome in ${GenomeLength}; do
		sed -i "s/^N .*$/N ${genome}/" "${FILENAME}"
		for num in $(seq 1 ${RUNALL}); do
			echo "GEN_MAX ${genome} POP_SIZE ${popsize} RUN ${num}"
			./onemax >> "${CWD}/fitnesstrend_${DATETIME}_${popsize}_${genome}_${num}_CPU.csv"
		done
	done
done
