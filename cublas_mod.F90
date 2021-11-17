module cublas_mod
!
! Define the INTERFACE to the NVIDIA C code cublasSgemm and cublasDgemm
!
interface cuda_gemm
!
! void cublasSgemm (char transa, char transb, int m, int n,
! int k, float alpha, const float *A, int lda,
! const float *B, int ldb, float beta, float *C, int ldc)
!
subroutine cuda_sgemm(cta, ctb, m, n, k,&
alpha, A, lda, B, ldb, beta, c, ldc) bind(C,name='cublasSgemm')
use iso_c_binding
character(1,c_char),value :: cta, ctb
integer(c_int),value :: m,n,k,lda,ldb,ldc
real(c_float),value :: alpha,beta
real(c_float), dimension(lda,*) :: A
real(c_float), dimension(ldb,*) :: B
real(c_float), dimension(ldc,*) :: C
end subroutine cuda_sgemm

!
! void cublasDgemm (char transa, char transb, int m, int n,
! int k, double alpha, const double *A, int lda,
! const double *B, int ldb, double beta, double *C, int ldc)
!
subroutine cuda_dgemm(cta, ctb, m, n, k,&
alpha, A, lda, B, ldb, beta, c, ldc) bind(C,name='cublasDgemm')
use iso_c_binding
character(1,c_char),value :: cta, ctb
integer(c_int),value :: m,n,k,lda,ldb,ldc
real(c_double),value :: alpha,beta
real(c_double), dimension(lda,*) :: A
real(c_double), dimension(ldb,*) :: B
real(c_double), dimension(ldc,*) :: C
end subroutine cuda_dgemm
end interface 

interface 
subroutine cuda_dgemm_batched(cta, ctb, m, n, k,&
alpha, A, lda, tda, B, ldb, tdb, beta, c, ldc, tdc, batchCount) bind(C,name='cublasDgemmBatched_wrapper')
use iso_c_binding
character(1,c_char),value :: cta, ctb
integer(c_int),value :: m,n,k,lda,ldb,ldc,tda,tdb,tdc,batchCount
real(c_double),value :: alpha,beta
real(c_double), dimension(lda,*) :: A
real(c_double), dimension(ldb,*) :: B
real(c_double), dimension(ldc,*) :: C
end subroutine cuda_dgemm_batched

subroutine cuda_dgemm_batched_finalize() bind(C,name='cublasDgemmBatched_finalize')
end subroutine cuda_dgemm_batched_finalize

end interface

interface

subroutine cuda_sgemm_batched(cta, ctb, m, n, k,&
alpha, A, lda, tda, B, ldb, tdb, beta, c, ldc, tdc, batchCount) bind(C,name='cublasSgemmBatched_wrapper')
use iso_c_binding
character(1,c_char),value :: cta, ctb
integer(c_int),value :: m,n,k,lda,ldb,ldc,tda,tdb,tdc,batchCount
real(c_float),value :: alpha,beta
real(c_float), dimension(lda,*) :: A
real(c_float), dimension(ldb,*) :: B
real(c_float), dimension(ldc,*) :: C
end subroutine cuda_sgemm_batched

subroutine cuda_sgemm_strided_batched(cta, ctb, m, n, k,&
alpha, A, lda, tda, B, ldb, tdb, beta, c, ldc, tdc, batchCount) bind(C,name='cublasSgemmStridedBatched_wrapper')
use iso_c_binding
character(1,c_char),value :: cta, ctb
integer(c_int),value :: m,n,k,lda,ldb,ldc,batchCount
integer(c_long_long),value :: tda,tdb,tdc
real(c_float),value :: alpha,beta
real(c_float), dimension(lda,*) :: A
real(c_float), dimension(ldb,*) :: B
real(c_float), dimension(ldc,*) :: C
end subroutine cuda_sgemm_strided_batched

subroutine cuda_sgemm_batched_finalize() bind(C,name='cublasSgemmBatched_finalize')
end subroutine cuda_sgemm_batched_finalize

end interface


end module cublas_mod
