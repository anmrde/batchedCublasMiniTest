#!/bin/bash 
nvcc cublasSgemmBatched.cu -dc -ccbin nvc++
nvcc cublasDgemmBatched.cu -dc -ccbin nvc++
nvfortran -c parkind1.F90
nvfortran -c cuda_device_mod.F90
nvfortran -c cublas_mod.F90
nvfortran -c cuda_gemm_batched_mod.F90 cublas_mod.o cuda_device_mod.o parkind1.o
nvfortran -o cublas-batch.exe cublas-batch.f90 cublas_mod.o cuda_device_mod.o parkind1.o cuda_gemm_batched_mod.o cublasDgemmBatched.o cublasSgemmBatched.o -acc -ta=nvidia -Minfo -Mcuda