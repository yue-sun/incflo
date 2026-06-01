#include <incflo.H>

using namespace amrex;

void incflo::compute_cp (int lev, MultiFab& cp) const
{
    bool const conservative_temperature = !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;
    auto& ld = *m_leveldata[lev];

    // Initialize all cells (including ghosts) to a safe baseline so AMR
    // coarse-fine interpolation and implicit solves never see undefined cp.
    cp.setVal(m_cp);

    int temp_iface_smooth_iters = 0;
    Real temp_iface_smooth_weight = Real(0.0);
    get_temperature_property_smoothing(temp_iface_smooth_iters, temp_iface_smooth_weight);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(cp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox();
        Array4<Real> const& cp_a = cp.array(mfi);

        if (!conservative_temperature) {
            Real l_cp = m_cp;
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                cp_a(i,j,k) = l_cp;
            });
            continue;
        }

#ifdef INCFLO_SIM_CRYO
        auto const& ct = ld.cell_type.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (ct(i,j,k) == -2) {
                cp_a(i,j,k) = cp_tcp;
            } else if (ct(i,j,k) == -3) {
                cp_a(i,j,k) = cp_plu;
            } else if (ct(i,j,k) == -4) {
                cp_a(i,j,k) = cp_sap;
            } else if (ct(i,j,k) == -5) {
                cp_a(i,j,k) = cp_dia;
            } else if (ct(i,j,k) == -6) {
                cp_a(i,j,k) = cp_plu;
            } else if (ct(i,j,k) == -1) {
                cp_a(i,j,k) = cp_eth;
            } else if (ct(i,j,k) >= 0) {
                cp_a(i,j,k) = cp_sam;
            } else {
                cp_a(i,j,k) = m_cp;
            }
        });
#else
        Real l_cp = m_cp;
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cp_a(i,j,k) = l_cp;
        });
#endif
    }

    smooth_temperature_property_at_interfaces(lev, cp,
                                              temp_iface_smooth_iters,
                                              temp_iface_smooth_weight);
}

void incflo::get_temperature_property_smoothing (int& smooth_iters,
                                                 Real& smooth_weight) const
{
    smooth_iters = 0;
    smooth_weight = Real(0.0);

    ParmParse pp("incflo");
    pp.query("temp_iface_smooth_iters", smooth_iters);
    pp.query("temp_iface_smooth_weight", smooth_weight);

    smooth_iters = amrex::max(0, smooth_iters);
    smooth_weight = amrex::min(Real(1.0),
                               amrex::max(Real(0.0), smooth_weight));
}

void incflo::smooth_temperature_property_at_interfaces (int lev,
                                                        MultiFab& property,
                                                        int smooth_iters,
                                                        Real smooth_weight) const
{
    property.FillBoundary(geom[lev].periodicity());

    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;
    if (!(conservative_temperature && smooth_iters > 0 && smooth_weight > Real(0.0))) {
        return;
    }

    auto& ct_mf = m_leveldata[lev]->cell_type;
    ct_mf.FillBoundary(geom[lev].periodicity());

    auto const domlo = lbound(geom[lev].Domain());
    auto const domhi = ubound(geom[lev].Domain());
    constexpr Real property_floor = Real(1.0e-300);

    MultiFab property_old(property.boxArray(), property.DistributionMap(), 1,
                          property.nGrowVect(), MFInfo(), property.Factory());

    for (int iter = 0; iter < smooth_iters; ++iter)
    {
        MultiFab::Copy(property_old, property, 0, 0, 1, property.nGrowVect());
        property_old.FillBoundary(geom[lev].periodicity());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(property, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& property_a = property.array(mfi);
            Array4<Real const> const& property0 = property_old.const_array(mfi);
            Array4<int const> const& ct = ct_mf.const_array(mfi);
            Real const w = smooth_weight;

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
                    property_a(i,j,k) = property0(i,j,k);
                    return;
                }

                Real l0 = std::log(amrex::max(property0(i,j,k), property_floor));
                Real lsum = l0;
                int nsum = 1;

                if (i > domlo.x) { lsum += std::log(amrex::max(property0(i-1,j,k), property_floor)); ++nsum; }
                if (i < domhi.x) { lsum += std::log(amrex::max(property0(i+1,j,k), property_floor)); ++nsum; }
                if (j > domlo.y) { lsum += std::log(amrex::max(property0(i,j-1,k), property_floor)); ++nsum; }
                if (j < domhi.y) { lsum += std::log(amrex::max(property0(i,j+1,k), property_floor)); ++nsum; }
#if (AMREX_SPACEDIM == 3)
                if (k > domlo.z) { lsum += std::log(amrex::max(property0(i,j,k-1), property_floor)); ++nsum; }
                if (k < domhi.z) { lsum += std::log(amrex::max(property0(i,j,k+1), property_floor)); ++nsum; }
#endif

                Real lavg = lsum / Real(nsum);
                property_a(i,j,k) = std::exp((Real(1.0)-w)*l0 + w*lavg);
            });
        }

        property.FillBoundary(geom[lev].periodicity());
    }
}