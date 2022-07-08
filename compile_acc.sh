#!/bin/bash 
nvcc cublasSgemmBatched.cu -dc
nvfortran -c cublas_mod.F90
nvfortran -Dacc -o cublas-batch.exe cublas-batch.F90 cublas_mod.o cublasSgemmBatched.o -acc -gpu=cc80 -Minfo=all -cuda -cudalib=cublas