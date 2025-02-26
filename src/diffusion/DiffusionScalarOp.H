#ifndef INCFLO_DIFF_SCALAR_OP_H_
#define INCFLO_DIFF_SCALAR_OP_H_

#include <AMReX_MLMG.H>

#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#endif
#include <AMReX_MLABecLaplacian.H>

class incflo;

class DiffusionScalarOp
{
public:
    DiffusionScalarOp (incflo* a_incflo);

    void diffuse_scalar (amrex::Vector<amrex::MultiFab*> const& tracer,
                         amrex::Vector<amrex::MultiFab*> const& density,
                         amrex::Vector<amrex::MultiFab const*> const& eta,
                         amrex::Real dt);

    void diffuse_vel_components (
                         amrex::Vector<amrex::MultiFab*> const& vel,
                         amrex::Vector<amrex::MultiFab*> const& density,
                         amrex::Vector<amrex::MultiFab const*> const& eta,
                         amrex::Real dt);

    void compute_laps (amrex::Vector<amrex::MultiFab*> const& laps,
                       amrex::Vector<amrex::MultiFab const*> const& a_scalar,
                       amrex::Vector<amrex::MultiFab const*> const& eta);

    void compute_divtau (amrex::Vector<amrex::MultiFab*> const& a_divtau,
                         amrex::Vector<amrex::MultiFab const*> const& a_vel,
                         amrex::Vector<amrex::MultiFab const*> const& density,
                         amrex::Vector<amrex::MultiFab const*> const& eta);

private:

    void readParameters ();

    incflo* m_incflo;

#ifdef AMREX_USE_EB
    std::unique_ptr<amrex::MLEBABecLap> m_eb_scal_solve_op;
    std::unique_ptr<amrex::MLEBABecLap> m_eb_scal_apply_op;
    std::unique_ptr<amrex::MLEBABecLap> m_eb_vel_solve_op;
    std::unique_ptr<amrex::MLEBABecLap> m_eb_vel_apply_op;
#endif
    std::unique_ptr<amrex::MLABecLaplacian> m_reg_scal_solve_op;
    std::unique_ptr<amrex::MLABecLaplacian> m_reg_scal_apply_op;
    std::unique_ptr<amrex::MLABecLaplacian> m_reg_vel_solve_op;
    std::unique_ptr<amrex::MLABecLaplacian> m_reg_vel_apply_op;

    // DiffusionOp verbosity
    int m_verbose = 0;

    // Options to control MLMG behavior
    int m_mg_verbose = 0;
    int m_mg_bottom_verbose = 0;
    int m_mg_max_iter = 100;
    int m_mg_bottom_maxiter = 100;
    int m_mg_max_fmg_iter = 0;
    int m_mg_max_coarsening_level = 100;
    int m_mg_maxorder = 2;

    int m_num_pre_smooth = 2;
    int m_num_post_smooth = 2;

#ifdef AMREX_USE_FLOAT
    amrex::Real m_mg_rtol = amrex::Real(1.0e-4);
    amrex::Real m_mg_atol = amrex::Real(1.0e-7);
#else
    amrex::Real m_mg_rtol = amrex::Real(1.0e-11);
    amrex::Real m_mg_atol = amrex::Real(1.0e-14);
#endif
    std::string m_mg_bottom_solver = "bicgstab";
};

#endif
