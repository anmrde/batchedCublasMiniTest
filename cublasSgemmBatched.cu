//
// Wrapper for cublasSgemm function.
//
// Alan Gray, NVIDIA
//

#include <stdio.h>
#include "cublas_v2.h"

bool alreadyAllocated_sgemm = false;
bool alreadyAllocated_sgemm_handle = false;

float **d_Aarray_sgemm;
float **d_Barray_sgemm;
float **d_Carray_sgemm;

float **Aarray_sgemm;
float **Barray_sgemm;
float **Carray_sgemm;

cublasHandle_t handle_sgemm;

extern "C" void cublasSgemmBatched_wrapper(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, int tda, const float *B, int ldb, int tdb, float beta, float *C, int ldc, int tdc, int batchCount)
{

  printf("CUBLAS m=%d,n=%d,k=%d,batchcount=%d\n", m, n, k, batchCount);

  cublasOperation_t op_t1 = CUBLAS_OP_N, op_t2 = CUBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = CUBLAS_OP_T;

  if (transb == 'T' || transb == 't')
    op_t2 = CUBLAS_OP_T;

  //float **Aarray_sgemm = (float**) malloc(batchCount*sizeof(float*));
  //float **Barray_sgemm = (float**) malloc(batchCount*sizeof(float*));
  //float **Carray_sgemm = (float**) malloc(batchCount*sizeof(float*));

  if (!alreadyAllocated_sgemm_handle)
  {
    if (cublasCreate(&handle_sgemm) == CUBLAS_STATUS_SUCCESS)
    {
      printf("Cuda in cublasSgemmBatched.cu: cublasCreate succeeded\n");
    }
    printf("after cublasCreate\n");
    printf("after cublasCreate: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
    alreadyAllocated_sgemm_handle = true;
  }

  if (!alreadyAllocated_sgemm)
  {
    if (cudaMallocHost(&Aarray_sgemm, batchCount * sizeof(float *)) == cudaSuccess)
    {
      printf("Cuda in cublasSgemmBatched.cu: cudaMallocHost A succeeded\n");
    }
    printf("after cudaMallocHost A\n");
    printf("after cudaMallocHost A: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
    if (cudaMallocHost(&Barray_sgemm, batchCount * sizeof(float *)) == cudaSuccess)
    {
      printf("Cuda in cublasSgemmBatched.cu: cudaMallocHost B succeeded\n");
    }
    printf("after cudaMallocHost B\n");
    printf("after cudaMallocHost B: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
    if (cudaMallocHost(&Carray_sgemm, batchCount * sizeof(float *)) == cudaSuccess)
    {
      printf("Cuda in cublasSgemmBatched.cu: cudaMallocHost C succeeded\n");
    }
    printf("after cudaMallocHost C\n");
    printf("after cudaMallocHost C: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
    alreadyAllocated_sgemm = true;
  }

  if (cudaMalloc(&d_Aarray_sgemm, batchCount * sizeof(float *)) == cudaSuccess)
  {
    printf("Cuda in cublasSgemmBatched.cu: cudaMalloc A succeeded\n");
  }
  printf("after cudaMalloc A\n");
  printf("after cudaMalloc A: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  if (cudaMalloc(&d_Barray_sgemm, batchCount * sizeof(float *)) == cudaSuccess)
  {
    printf("Cuda in cublasSgemmBatched.cu: cudaMalloc A succeeded\n");
  }
  printf("after cudaMalloc B\n");
  printf("after cudaMalloc B: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  if (cudaMalloc(&d_Carray_sgemm, batchCount * sizeof(float *)) == cudaSuccess)
  {
    printf("Cuda in cublasSgemmBatched.cu: cudaMalloc A succeeded\n");
  }
  printf("after cudaMalloc C\n");
  printf("after cudaMalloc C: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  int i;
  for (i = 0; i < batchCount; i++)
  {
    Aarray_sgemm[i] = (float *)&(A[i * lda * tda]);
    Barray_sgemm[i] = (float *)&(B[i * ldb * tdb]);
    Carray_sgemm[i] = (float *)&(C[i * ldc * tdc]);
  }
  if (cudaMemcpy(d_Aarray_sgemm, Aarray_sgemm, batchCount * sizeof(float *), cudaMemcpyHostToDevice) == cudaSuccess)
  {
    printf("Cuda in cublasSgemmBatched.cu: cudaMemcpy A succeeded\n");
  }
  printf("after cudaMemcpy A\n");
  printf("after cudaMemcpy A: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  if (cudaMemcpy(d_Barray_sgemm, Barray_sgemm, batchCount * sizeof(float *), cudaMemcpyHostToDevice) == cudaSuccess)
  {
    printf("Cuda in cublasSgemmBatched.cu: cudaMemcpy B succeeded\n");
  }
  printf("after cudaMemcpy B\n");
  printf("after cudaMemcpy B: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  if (cudaMemcpy(d_Carray_sgemm, Carray_sgemm, batchCount * sizeof(float *), cudaMemcpyHostToDevice) == cudaSuccess)
  {
    printf("Cuda in cublasSgemmBatched.cu: cudaMemcpy C succeeded\n");
  }
  printf("after cudaMemcpy C\n");
  printf("before sgemm: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  if (cublasSgemmBatched(handle_sgemm, op_t1, op_t2, m, n, k, &alpha, (const float **)d_Aarray_sgemm, lda, (const float **)d_Barray_sgemm, ldb, &beta, (float **)d_Carray_sgemm, ldc, batchCount) == CUBLAS_STATUS_SUCCESS)
  {
    printf("Cuda in cublasSgemmBatched.cu: cublasSgemmBatched succeeded\n");
  }
  printf("after sgemm: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  printf("after sgemm\n");
  int syncres = cudaDeviceSynchronize();
  printf("cublasSgemmBatched.cu: cudaDeviceSynchronize() returns %d\n", syncres);
  //cudaDeviceSynchronize();
  if (syncres != cudaSuccess)
  {
    fprintf(stderr, "Cuda error 2 in cublasSgemmBatched.cu: Failed to synchronize\n");
    return;
  }

  //cudaFree(Aarray_sgemm);
  //cudaFree(Barray_sgemm);
  //cudaFree(Carray_sgemm);

  cudaFree(d_Aarray_sgemm);
  cudaFree(d_Barray_sgemm);
  cudaFree(d_Carray_sgemm);
  //cublasDestroy(handle_sgemm);
}

extern "C" void cublasSgemmStridedBatched_wrapper(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, long long tda, const float *B, int ldb, long long tdb, float beta, float *C, int ldc, long long tdc, int batchCount)
{

  // printf("CUBLAS m=%d,n=%d,k=%d,batchcount=%d\n",m,n,k,batchCount);

  cublasOperation_t op_t1 = CUBLAS_OP_N, op_t2 = CUBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = CUBLAS_OP_T;

  if (transb == 'T' || transb == 't')
    op_t2 = CUBLAS_OP_T;

  if (!alreadyAllocated_sgemm_handle)
  {
    cublasCreate(&handle_sgemm);
    alreadyAllocated_sgemm_handle = true;
  }
  cublasSgemmStridedBatched(handle_sgemm, op_t1, op_t2, m, n, k, &alpha, (const float *)A, lda, tda, (const float *)B, ldb, tdb, &beta, (float *)C, ldc, tdc, batchCount);
}

extern "C" void cublasSgemmBatched_finalize()
{

  if (alreadyAllocated_sgemm)
  {

    cudaFree(Aarray_sgemm);
    cudaFree(Barray_sgemm);
    cudaFree(Carray_sgemm);

    cudaFree(d_Aarray_sgemm);
    cudaFree(d_Barray_sgemm);
    cudaFree(d_Carray_sgemm);
  }

  if (alreadyAllocated_sgemm_handle)
  {
    cublasDestroy(handle_sgemm);
  }
}