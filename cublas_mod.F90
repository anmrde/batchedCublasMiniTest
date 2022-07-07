module cublas_mod

interface

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
