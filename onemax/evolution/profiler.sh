#!/bin/bash

OUTFILE=${1}
nsys profile -t cuda,osrt,nvtx,cudnn,cublas \
		 --stats=true                   \
		 -f true                        \
	   -o ${OUTFILE} \
		 -w true ./gpuonemax
