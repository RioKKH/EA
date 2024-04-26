#!/bin/bash

readonly NUMOFRUN=30

main()
{
	for _ in $(seq 1 1 "${NUMOFRUN}"); do
		./onemax
		#./gpuonemax | grep -v "End"
	done
}

main
