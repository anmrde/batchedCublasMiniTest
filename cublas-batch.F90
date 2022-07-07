program main
    USE CUBLAS_MOD
#ifdef acc
    USE openacc
#else
    USE OMP_LIB
#endif
    USE ISO_C_BINDING
    use ieee_arithmetic
    implicit none
 
    integer(KIND=c_int) :: R_NDGNH,IF_FS_INV,IF_FS_INV0,R_NTMAX,R_NSMAX,ITDZCA, &
        & ILDZBS,ITDZBA,LDZAA,TDZAS,ILDZCS,D_NUMP
    integer(KIND=c_int) :: ILDZBA,TDZAA,ILDZCA
    integer(KIND=c_long_long) :: ILDZBA2,TDZAA2,ILDZCA2
    real(KIND=c_float),dimension(:,:,:), allocatable :: IZBS,IZCST
    real(KIND=c_float),dimension(:,:,:), allocatable :: ZAA
 
    R_NDGNH = 80
    IF_FS_INV = 4
    IF_FS_INV0 = 1924
    R_NTMAX = 79
    R_NSMAX = 79
    ITDZCA = IF_FS_INV
    ILDZCA = R_NDGNH
    ILDZBA = (R_NSMAX+2)/2
    ILDZBS = (R_NSMAX+3)/2
    ITDZBA = IF_FS_INV
    LDZAA = R_NDGNH
    TDZAA = (R_NTMAX+2)/2
    TDZAS = (R_NTMAX+3)/2
    ILDZCS = ILDZCA
    D_NUMP = R_NDGNH
    ILDZBA2 = ILDZBA
    TDZAA2 = TDZAA
    ILDZCA2 = ILDZCA

    ! Allocate host storage for IZBS,ZAA,IZCST matrices
    allocate(IZBS(IF_FS_INV0,ILDZBS,D_NUMP))
    allocate(ZAA(LDZAA,TDZAA,D_NUMP))
    allocate(IZCST(IF_FS_INV0,ILDZCS,D_NUMP))
    
#ifdef acc
    !$acc enter data create(IZBS,ZAA,IZCST)
#else
    !$omp target enter data map(alloc:IZBS,ZAA,IZCST)
#endif
 
    IZBS = 1._c_float
#ifdef acc
    !$acc update device(IZBS)
#else
    !$omp target update to(IZBS)
#endif
    ZAA = 2._c_float
#ifdef acc
    !$acc update device(ZAA)
#else
    !$omp target update to(ZAA)
#endif
    IZCST = 0._c_float
#ifdef acc
    !$acc update device(IZCST)
#else
    !$omp target update to(IZCST)
#endif

#ifdef acc
    !$acc data present(IZBS,ZAA,IZCST)
#else
    !Anything equivalent? Intel's migration tool suggested the following code which doesn't compile:
    !!$omp target data map(present,alloc:izbs,zaa,izcst)
#endif

#ifdef acc
    !$acc host_data use_device(IZBS,ZAA,IZCST)
#else
    !$omp target data use_device_addr(IZBS,ZAA,IZCST)
    !!$omp target data use_device_ptr(IZBS,ZAA,IZCST)
#endif
    CALL CUDA_SGEMM_STRIDED_BATCHED('N','T',ITDZCA,ILDZCA,ILDZBA,1._c_float,IZBS,ITDZBA,ILDZBA2,&
          & ZAA,LDZAA,TDZAA2,0._c_float,IZCST,ITDZCA,ILDZCA2,D_NUMP)
#ifdef acc
    !$acc end host_data
    !$acc update host(IZCST)
#else
    !$omp target update from(IZCST)
#endif
    print*,'IZCST: sum=',SUM(IZCST(:,:,:))
#ifdef acc
    !$acc end data
    !$acc exit data delete(IZBS,ZAA,IZCST)
#else
    !$omp end target data
    !$omp target exit data map(delete:IZBS,ZAA,IZCST)
#endif

    deallocate(IZBS)
    deallocate(ZAA)
    deallocate(IZCST)

end program