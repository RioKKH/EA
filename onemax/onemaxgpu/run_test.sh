#!/bin/bash

readonly CWD=$(pwd)
readonly PARAMSFILE=${CWD}/onemax.prms

readonly POPULATION=$(seq 32 32 1024)
readonly CHROMOSOME=$(seq 32 32 1024)

for pop in ${POPULATION}; do
	sed -i "s/^POPSIZE.*$/POPSIZE                   ${pop}/" ${PARAMSFILE}
	for chr in ${CHROMOSOME}; do
		echo ${pop} ${chr}
		sed -i "s/^CHROMOSOME.*$/CHROMOSOME                ${chr}/" ${PARAMSFILE}
		./onemax
	done
done
