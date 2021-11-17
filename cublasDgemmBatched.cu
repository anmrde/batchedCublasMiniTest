//
// Wrapper for cublasDgemm function. 
//
// Alan Gray, NVIDIA
//

#include <stdio.h>
#include "cublas_v2.h" 


bool alreadyAllocated=false;

double **d_Aarray;
double **d_Barray;
double **d_Carray;

double **Aarray;
double **Barray;
double **Carray;

cublasHandle_t handle;	

extern "C" void cublasDgemmBatched_wrapper (char transa, char transb, int m, int n,int k, double alpha, const double *A, int lda, int tda, const double *B, int ldb, int tdb, double beta, double *C, int ldc, int tdc, int batchCount)
{


  // printf("CUBLAS m=%d,n=%d,k=%d,batchcount=%d\n",m,n,k,batchCount);

 
  cublasOperation_t op_t1=CUBLAS_OP_N, op_t2=CUBLAS_OP_N;

  if (transa=='T' || transa=='t')		
    op_t1=CUBLAS_OP_T;

  if (transb=='T' || transb=='t')		
    op_t2=CUBLAS_OP_T;


  //double **Aarray = (double**) malloc(batchCount*sizeof(double*));
  //double **Barray = (double**) malloc(batchCount*sizeof(double*));
  //double **Carray = (double**) malloc(batchCount*sizeof(double*));


  if (!alreadyAllocated){

    cublasCreate(&handle);

    cudaMallocHost(&Aarray,batchCount*sizeof(double*));
    cudaMallocHost(&Barray,batchCount*sizeof(double*));
    cudaMallocHost(&Carray,batchCount*sizeof(double*));
        
    cudaMalloc(&d_Aarray,batchCount*sizeof(double*));
    cudaMalloc(&d_Barray,batchCount*sizeof(double*));
    cudaMalloc(&d_Carray,batchCount*sizeof(double*));
 
    alreadyAllocated=true;
  }

  int i;
  for(i=0;i<batchCount;i++){
    Aarray[i]=(double*) &(A[i*lda*tda]);
    Barray[i]=(double*) &(B[i*ldb*tdb]);
    Carray[i]=(double*) &(C[i*ldc*tdc]);
  }

  cudaMemcpy(d_Aarray,Aarray,batchCount*sizeof(double*),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Barray,Barray,batchCount*sizeof(double*),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Carray,Carray,batchCount*sizeof(double*),cudaMemcpyHostToDevice);


  cublasDgemmBatched(handle,op_t1,op_t2,m,n,k,&alpha,(const double**) d_Aarray,lda, (const double**) d_Barray,ldb,&beta,(double**) d_Carray,ldc,batchCount);

  //printf("after dgemm\n");
  cudaDeviceSynchronize();
  
  //cudaFree(Aarray);
  //cudaFree(Barray);
  //cudaFree(Carray);
  
  //cudaFree(d_Aarray);
  //cudaFree(d_Barray);
  //cudaFree(d_Carray);
  //cublasDestroy(handle);
  
  
}

extern "C" void cublasDgemmBatched_finalize ()
{



  if (alreadyAllocated){
  
    cudaFree(Aarray);
    cudaFree(Barray);
    cudaFree(Carray);
    
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
    cudaFree(d_Carray);
    cublasDestroy(handle);

  }
  
}
