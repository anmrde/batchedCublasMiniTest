! most of this code originates from a code example at
! https://forums.developer.nvidia.com/t/openacc-with-cublas-batched-routine-in-fortran/135158
program main
    USE CUDA_GEMM_BATCHED_MOD
    USE CUDA_DEVICE_MOD
    USE PARKIND1

    USE openacc
    USE ISO_C_BINDING
    use ieee_arithmetic
    implicit none
 
    integer(KIND=JPIM) :: dim, stat, i, j, k, batch_count, stride
    real(KIND=JPRM),dimension(:,:,:), allocatable :: A, B, C
    real(KIND=JPRM) :: alpha, beta, index, sum
 
    !Linear dimension of matrices
    dim = 100

    stride = 0
 
    ! Number of A,B,C matrix sets
    batch_count = 1000
 
    ! Allocate host storage for A,B,C square matrices
    allocate(A(dim,dim,batch_count))
    allocate(B(dim,dim,batch_count))
    allocate(C(dim,dim,batch_count))
    
    !$acc enter data create(A,B,C)
 
 
    ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
    ! Matrices are arranged column major

!$acc data present(A,B,C)
!$acc kernels
!$acc loop
    do k=1,batch_count
!$acc loop
        do j=1,dim
!$acc loop
            do i=1,dim
                if (i==j) then
                    index = real(j*dim + i)
                    a(i,j,k) = k*sin(index)
                    b(i,j,k) = sin(index)
                    c(i,j,k) = k*cos(index) * cos(index)
                else
                    a(i,j,k) = 0.0
                    b(i,j,k) = 0.0
                    c(i,j,k) = 0.0
                endif
            enddo ! i
        enddo ! j
    enddo ! k

!$acc end kernels
 

    ! Set matrix coefficients
    alpha = 1.0
    beta = 1.0
 
    ! batched DGEMM: C = alpha*A*B + beta*C

!$acc host_data use_device(A,B,C)
    CALL CUDA_GEMM_BATCHED('N','N',dim,dim,dim,alpha,&
        A,dim,stride,B,dim,stride,beta,C,dim,stride,batch_count)
!$acc end host_data

    ! Simple sanity test, sum up all elements
    sum = 0.0
!$acc kernels
!$acc loop
    do k=1,batch_count
!$acc loop
        do j=1,dim
            do i=1,dim
                sum = sum + C(i,j,k)
            enddo
        enddo
    enddo
!$acc end kernels
    !print *, "Sum is:", sum, "should be: ", dim*(batch_count)*(batch_count+1)/2

!$acc end data
 
!$acc exit data delete(A,B,C)
    deallocate(A)
    deallocate(B)
    deallocate(C)

end program