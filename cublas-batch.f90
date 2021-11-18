program main
    USE CUDA_GEMM_BATCHED_MOD
    USE CUDA_DEVICE_MOD
    USE PARKIND1

    USE openacc
    USE ISO_C_BINDING
    use ieee_arithmetic
    implicit none
 
    integer(KIND=JPIM) :: R_NDGNH,IF_FS_INV,IF_FS_INV0,R_NTMAX,R_NSMAX,ITDZCA, &
        & ILDZBS,ITDZBA,LDZAA,TDZAS,ILDZCS,D_NUMP
    integer(KIND=JPIM) :: ILDZBA,TDZAA,ILDZCA
    integer(KIND=JPIM) :: ILDZBA2,TDZAA2,ILDZCA2 ! this should be KIND=JPIB, but somehow it doesn't work then
    real(KIND=JPRM),dimension(:), allocatable :: IZBS,IZCST
    real(KIND=JPRM),dimension(:,:,:), allocatable :: ZAA
 
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
    allocate(IZBS(IF_FS_INV0*ILDZBS*D_NUMP))
    allocate(ZAA(LDZAA,TDZAA,D_NUMP))
    allocate(IZCST(IF_FS_INV0*ILDZCS*D_NUMP))
    
    !$acc enter data create(IZBS,ZAA,IZCST)
 
    IZBS = 0._JPRM
    !$acc update device(IZBS)
    ZAA = 0._JPRM
    !$acc update device(ZAA)
    IZCST = 0._JPRM
    !$acc update device(IZCST)

    ! printing parameters to check against full simulation:
    print*,'ITDZCA,ILDZCA,ILDZBA=',ITDZCA,ILDZCA,ILDZBA
    print*,'shape(IZBS)=',shape(IZBS)
    print*,'ITDZBA,ILDZBA=',ITDZBA,ILDZBA
    print*,'shape(ZAA)=',shape(ZAA)
    print*,'LDZAA,TDZAA=',LDZAA,TDZAA
    print*,'shape(IZCST)=',shape(IZCST)
    print*,'ITDZCA,ILDZCA,D_NUMP=',ITDZCA,ILDZCA,D_NUMP
    
    !$acc data present(IZBS,ZAA,IZCST)

    !$acc host_data use_device(IZBS,ZAA,IZCST)
    CALL CUDA_GEMM_BATCHED('N','T',ITDZCA,ILDZCA,ILDZBA,1.0_JPRM,IZBS,ITDZBA,ILDZBA2,&
          & ZAA,LDZAA,TDZAA2,0._JPRM,IZCST,ITDZCA,ILDZCA2,D_NUMP)
    !$acc end host_data

    !$acc end data

    !$acc exit data delete(IZBS,ZAA,IZCST)
    deallocate(IZBS)
    deallocate(ZAA)
    deallocate(IZCST)

end program