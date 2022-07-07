#!/bin/bash 
nvcc cublasSgemmBatched.cu -dc -ccbin nvc++
nvfortran -c cublas_mod.F90
nvfortran -o cublas-batch.exe cublas-batch.F90 cublas_mod.o cublasSgemmBatched.o -mp=gpu -ta=nvidia -Minfo -Mcuda