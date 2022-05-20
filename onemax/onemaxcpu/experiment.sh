#!/bin/bash

CWD=$(pwd)
FILENAME=onemax_prms.dat
ORIGINAL=${FILENAME}.org
GenomeLength=$(seq 32 32 512)
PopulationSize=$(seq 32 32 512)

CATLOG=${CWD}/cat.dat
RESULT=${CWD}/result3.dat

if [ ! -f ${ORIGINAL} ]; then
	cp -p ${FILENAME} ${ORIGINAL}
fi

for genome in ${GenomeLength}; do
	for popsize in ${PopulationSize}; do
		sed -i "s/^N .*$/N ${genome}/" ${FILENAME}
		# sed -i "s/^GEN_MAX.*$/GEN_MAX ${genome}/" ${FILENAME}
		sed -i "s/^POP_SIZE.*$/POP_SIZE ${popsize}/" ${FILENAME}
		cat ${FILENAME} 2>> ${CATLOG}
		./onemax >> ${RESULT}
		sleep 0.1
	done
done
