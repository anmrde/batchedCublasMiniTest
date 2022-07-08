#!/bin/bash 
nvcc cublasSgemmBatched.cu -dc
nvfortran -c cublas_mod.F90
nvfortran -o cublas-batch.exe cublas-batch.F90 cublas_mod.o cublasSgemmBatched.o -mp=gpu -gpu=cc80 -Minfo=all -cuda -cudalib=cublas