AMREX_HOME ?= ../../amrex
AMREX_HYDRO_HOME ?= ../../AMReX-Hydro
TOP = ..

EBASE = incflo

DIM = 3

PRECISION  = DOUBLE

TINY_PROFILE = FALSE

DEBUG      = FALSE

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = TRUE
CUDA_ARCH  = 86        # RTX A6000 = GA102 = sm_86 (must precede Make.incflo: Make.defs uses CUDA_ARCH ?= 70)

USE_HYPRE = FALSE
BL_NO_FORT = TRUE

USE_EB = FALSE

USE_CSG = FALSE

SIM_CRYO = TRUE

include ../src/Make.incflo
