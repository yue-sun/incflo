#include <incflo.H>
#include <incflo_derive_K.H>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real expterm (amrex::Real nu) noexcept
{
    return (nu < Real(1.e-9)) ? (Real(1.0)-Real(0.5)*nu+nu*nu*Real(1.0/6.0)-(nu*nu*nu)*Real(1./24.))
                        : -std::expm1(-nu)/nu;
}

struct NonNewtonianViscosity
{
    incflo::FluidModel fluid_model;
    amrex::Real mu, n_flow, tau_0, eta_0, papa_reg;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real sr) const noexcept {
        switch (fluid_model)
        {
        case incflo::FluidModel::powerlaw:
        {
            return mu * std::pow(sr,n_flow-Real(1.0));
        }
        case incflo::FluidModel::Bingham:
        {
            return mu + tau_0 * expterm(sr/papa_reg) / papa_reg;
        }
        case incflo::FluidModel::HerschelBulkley:
        {
            return (mu*std::pow(sr,n_flow)+tau_0)*expterm(sr/papa_reg)/papa_reg;
        }
        case incflo::FluidModel::deSouzaMendesDutra:
        {
            return (mu*std::pow(sr,n_flow)+tau_0)*expterm(sr*(eta_0/tau_0))*(eta_0/tau_0);
        }
        default:
        {
            return mu;
        }
        };
    }
};

}

void incflo::compute_viscosity (Vector<MultiFab*> const& vel_eta,
                                Vector<MultiFab*> const& rho,
                                Vector<MultiFab*> const& vel,
                                Real time, int nghost)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        compute_viscosity_at_level(lev, vel_eta[lev], rho[lev], vel[lev], geom[lev], time, nghost);
    }
}

#ifdef AMREX_USE_EB
void incflo::compute_viscosity_at_level (int lev,
#else
void incflo::compute_viscosity_at_level (int /*lev*/,
#endif
                                         MultiFab* vel_eta,
                                         MultiFab* /*rho*/,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         Real /*time*/, int nghost)
{
    if (m_fluid_model == FluidModel::Newtonian)
    {
        vel_eta->setVal(m_mu, 0, 1, nghost);
    }
    else
    {
        NonNewtonianViscosity non_newtonian_viscosity;
        non_newtonian_viscosity.fluid_model = m_fluid_model;
        non_newtonian_viscosity.mu = m_mu;
        non_newtonian_viscosity.n_flow = m_n_0;
        non_newtonian_viscosity.tau_0 = m_tau_0;
        non_newtonian_viscosity.eta_0 = m_eta_0;
        non_newtonian_viscosity.papa_reg = m_papa_reg;

#ifdef AMREX_USE_EB
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
#endif

        Real idx = Real(1.0) / lev_geom.CellSize(0);
        Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
        Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*vel_eta,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
                Box const& bx = mfi.growntilebox(nghost);
                Array4<Real> const& eta_arr = vel_eta->array(mfi);
                Array4<Real const> const& vel_arr = vel->const_array(mfi);
#ifdef AMREX_USE_EB
                auto const& flag_fab = flags[mfi];
                auto typ = flag_fab.getType(bx);
                if (typ == FabType::covered)
                {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        eta_arr(i,j,k) = Real(0.0);
                    });
                }
                else if (typ == FabType::singlevalued)
                {
                    auto const& flag_arr = flag_fab.const_array();
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real sr = incflo_strainrate_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr,flag_arr(i,j,k));
                        eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                    });
                }
                else
#endif
                {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real sr = incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr);
                        eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                    });
                }
        }
    }
}

void incflo::compute_tracer_diff_coeff (Vector<MultiFab*> const& tra_eta, int nghost) const
{
    for (auto *mf : tra_eta) {
        for (int n = 0; n < m_ntrac; ++n) {
            mf->setVal(m_mu_s[n], n, 1, nghost);
        }
    }
}

void incflo::compute_temperature_diff_coeff (Real /*time*/, Vector<MultiFab*> const& tem_eta) const
{
    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

    int temp_iface_smooth_iters = 0;
    Real temp_iface_smooth_weight = Real(0.0);
    {
        ParmParse pp("incflo");
        pp.query("temp_iface_smooth_iters", temp_iface_smooth_iters);
        pp.query("temp_iface_smooth_weight", temp_iface_smooth_weight);
    }
    temp_iface_smooth_iters = amrex::max(0, temp_iface_smooth_iters);
    temp_iface_smooth_weight = amrex::min(Real(1.0),
                                          amrex::max(Real(0.0), temp_iface_smooth_weight));

    for (int lev = 0; lev < static_cast<int>(tem_eta.size()); ++lev)
    {
        auto *mf = tem_eta[lev];
        // Outside cryo-material regions, use input value m_mu_T.
        mf->setVal(m_mu_T);

        auto const& ct_mf = m_leveldata[lev]->cell_type;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real>      const& eta = mf->array(mfi);
            Array4<int const> const& ct  = ct_mf.const_array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (ct(i, j, k) == -2) { // thermocouple
                    eta(i, j, k) = conservative_temperature ? k_tcp : kappa_tcp;
                } else if (ct(i, j, k) == -3) { // EM grid
                    eta(i, j, k) = conservative_temperature ? k_plu : kappa_plu;
                } else if (ct(i, j, k) == -4) { // sapphire disk
                    eta(i, j, k) = conservative_temperature ? k_sap : kappa_sap;
                } else if (ct(i, j, k) == -5) { // diamond disk
                    eta(i, j, k) = conservative_temperature ? k_dia : kappa_dia;
                } else if (ct(i, j, k) == -6) { // debug sphere (same as EM grid)
                    eta(i, j, k) = conservative_temperature ? k_plu : kappa_plu;
                } else if (ct(i, j, k) >= 0) { // samples
                    eta(i, j, k) = conservative_temperature ? k_sam : kappa_sam;
                } else if (ct(i, j, k) == -1) { // liquid ethane
                    eta(i, j, k) = conservative_temperature ? k_eth : kappa_eth;
                }
            });
        }

        // Optional narrow-band smoothing for conservative temperature runs.
        // This regularizes very sharp multi-material jumps and can damp interface spikes.
        if (conservative_temperature && temp_iface_smooth_iters > 0 &&
            temp_iface_smooth_weight > Real(0.0))
        {
            auto const domlo = lbound(geom[lev].Domain());
            auto const domhi = ubound(geom[lev].Domain());
            constexpr Real eta_floor = Real(1.0e-300);

            MultiFab eta_old(mf->boxArray(), mf->DistributionMap(), 1, mf->nGrowVect(),
                             MFInfo(), mf->Factory());

            for (int iter = 0; iter < temp_iface_smooth_iters; ++iter)
            {
                MultiFab::Copy(eta_old, *mf, 0, 0, 1, mf->nGrowVect());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& bx = mfi.tilebox();
                    Array4<Real> const& eta = mf->array(mfi);
                    Array4<Real const> const& eta0 = eta_old.const_array(mfi);
                    Array4<int const> const& ct  = ct_mf.const_array(mfi);
                    Real const w = temp_iface_smooth_weight;

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int c = ct(i,j,k);
                        bool interface_cell = false;

                        if (i > domlo.x && ct(i-1,j,k) != c) interface_cell = true;
                        if (i < domhi.x && ct(i+1,j,k) != c) interface_cell = true;
                        if (j > domlo.y && ct(i,j-1,k) != c) interface_cell = true;
                        if (j < domhi.y && ct(i,j+1,k) != c) interface_cell = true;
#if (AMREX_SPACEDIM == 3)
                        if (k > domlo.z && ct(i,j,k-1) != c) interface_cell = true;
                        if (k < domhi.z && ct(i,j,k+1) != c) interface_cell = true;
#endif

                        if (!interface_cell) {
                            eta(i,j,k) = eta0(i,j,k);
                            return;
                        }

                        Real l0 = std::log(amrex::max(eta0(i,j,k), eta_floor));
                        Real lsum = l0;
                        int nsum = 1;

                        if (i > domlo.x) { lsum += std::log(amrex::max(eta0(i-1,j,k), eta_floor)); ++nsum; }
                        if (i < domhi.x) { lsum += std::log(amrex::max(eta0(i+1,j,k), eta_floor)); ++nsum; }
                        if (j > domlo.y) { lsum += std::log(amrex::max(eta0(i,j-1,k), eta_floor)); ++nsum; }
                        if (j < domhi.y) { lsum += std::log(amrex::max(eta0(i,j+1,k), eta_floor)); ++nsum; }
#if (AMREX_SPACEDIM == 3)
                        if (k > domlo.z) { lsum += std::log(amrex::max(eta0(i,j,k-1), eta_floor)); ++nsum; }
                        if (k < domhi.z) { lsum += std::log(amrex::max(eta0(i,j,k+1), eta_floor)); ++nsum; }
#endif

                        Real lavg = lsum / Real(nsum);
                        eta(i,j,k) = std::exp((Real(1.0)-w)*l0 + w*lavg);
                    });
                }
            }
        }
    }
}

