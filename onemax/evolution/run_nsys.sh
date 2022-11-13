#!/bin/bash

nsys -profile -t cuda,osrt,nvtx,cudnn,cublas -o ${1} -w true ./gpuonemax
