#!/bin/bash

readonly CWD=$(pwd)
readonly PARAMSFILE=${CWD}/onemax.prms

readonly POPULATION=$(seq 32 32 1024)
readonly CHROMOSOME="32 64 128 256 512 1024"
readonly DATETIME=$(date +%Y%m%d-%H%M%S)
readonly BACKUPFILE=${CWD}/result_${DATETIME}.csv

readonly RESULTFILE=${CWD}/result.csv
if [[ -f ${RESULTFILE} ]]; then
	mv ${RESULTFILE} ${BACKUPFILE}
	rm ${RESULTFILE}
fi

for pop in ${POPULATION}; do
	sed -i "s/^POPSIZE.*$/POPSIZE                   ${pop}/" ${PARAMSFILE}
	for chr in ${CHROMOSOME}; do
		echo ${pop} ${chr}
		sed -i "s/^CHROMOSOME.*$/CHROMOSOME                ${chr}/" ${PARAMSFILE}
		./gpuonemax >> ${BACKUPFILE}
	done
done



