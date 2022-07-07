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

extern "C" void cublasSgemmStridedBatched_wrapper(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, long long tda, const float *B, int ldb, long long tdb, float beta, float *C, int ldc, long long tdc, int batchCount)
{

  printf("CUBLAS m=%d,n=%d,k=%d,batchcount=%d\n", m, n, k, batchCount);

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
  printf("before sgemm: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
  if (cublasSgemmStridedBatched(handle_sgemm, op_t1, op_t2, m, n, k, &alpha, (const float *)A, lda, tda, (const float *)B, ldb, tdb, &beta, (float *)C, ldc, tdc, batchCount) == CUBLAS_STATUS_SUCCESS)
  {
    printf("Cuda in cublasSgemmBatched.cu: cublasSgemmBatched succeeded\n");
  }
  printf("after sgemm: cudaDeviceSynchronize=%d\n", cudaDeviceSynchronize());
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