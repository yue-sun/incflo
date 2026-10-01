AMREX_HOME ?= ../../amrex
AMREX_HYDRO_HOME ?= ../../AMReX-Hydro
TOP = ..

EBASE = incflo

DIM = 3

PRECISION  = DOUBLE

TINY_PROFILE = FALSE

DEBUG      = FALSE

COMP	   = gnu

# Build switch: GPU=TRUE -> CUDA, GPU=FALSE -> CPU with MPI (+ OpenMP if OMP=TRUE).
# Defaults: Macs (no CUDA) build CPU/MPI, the cluster builds CUDA.
#   cluster GPU:          make
#   cluster MPI+OpenMP:   make GPU=FALSE OMP=TRUE
#   Mac MPI:              make
#   Mac MPI+OpenMP:       make OMP=TRUE
ifeq ($(shell uname -s),Darwin)
  GPU = FALSE
else
  GPU = TRUE
endif
OMP = FALSE

ifeq ($(GPU),TRUE)
  USE_MPI    = FALSE
  USE_OMP    = FALSE
  USE_CUDA   = TRUE
  # RTX A6000 = GA102 = sm_86 (must precede Make.incflo: Make.defs uses CUDA_ARCH ?= 70)
  CUDA_ARCH  = 86
else ifeq ($(GPU),FALSE)
  USE_MPI    = TRUE
  USE_OMP    = $(OMP)
  USE_CUDA   = FALSE
else
  $(error GPU must be TRUE or FALSE, got '$(GPU)')
endif

USE_HYPRE = FALSE
BL_NO_FORT = TRUE

USE_EB = FALSE

USE_CSG = FALSE

SIM_CRYO = TRUE

include ../src/Make.incflo
