#!/bin/bash

CWD=$(pwd)

RUNALL=15
FILENAME=onemax_prms.dat
ORIGINAL=${FILENAME}.org
GenomeLength=$(seq 32 32 1024)
PopulationSize=$(seq 32 32 1024)
DATETIME=$(date +%Y%m%d-%H%M%S)
readonly RESULTFILE=${CWD}/result_${DATETIME}.csv

CATLOG=${CWD}/cat.dat

if [ ! -f ${ORIGINAL} ]; then
	cp -p ${FILENAME} ${ORIGINAL}
fi

for genome in ${GenomeLength}; do
	for popsize in ${PopulationSize}; do
		sed -i "s/^N .*$/N ${genome}/" ${FILENAME}
		sed -i "s/^POP_SIZE.*$/POP_SIZE ${popsize}/" ${FILENAME}
		for _ in $(seq 1 ${RUNALL}); do
		#for num in $(seq 1 ${RUNALL}); do
			cat ${FILENAME} 2>> "${CATLOG}"
			./onemax >> "${RESULTFILE}"
			#./onemax >> "CPU_${popsize}_${genome}_${num}.csv"
		done
	done
done

# code backup
# sed -i "s/^GEN_MAX.*$/GEN_MAX ${genome}/" ${FILENAME}
