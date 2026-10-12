AMREX_HOME ?= ../../amrex
AMREX_HYDRO_HOME ?= ../../AMReX-Hydro
TOP = ..

EBASE = incflo

DIM = 3

PRECISION  = DOUBLE

TINY_PROFILE = FALSE

DEBUG      = FALSE

COMP	   = gnu

# Build switch: MPI=TRUE/FALSE, OMP=TRUE/FALSE. Default is MPI only (one OpenMP
# thread per rank), which runs faster than OpenMP on both the Mac and the server.
#   MPI:                  make
#   MPI+OpenMP:           make OMP=TRUE
#   OpenMP only (old):    make MPI=FALSE OMP=TRUE
# No CUDA: init_rfb_geometry reads m_rfb_fibers (host memory) inside ParallelFor.
MPI = TRUE
OMP = FALSE

USE_MPI    = $(MPI)
USE_OMP    = $(OMP)
USE_CUDA   = FALSE

USE_HYPRE = FALSE
BL_NO_FORT = TRUE

USE_EB = FALSE

USE_CSG = FALSE

SIM_RFB = TRUE

include ../src/Make.incflo
