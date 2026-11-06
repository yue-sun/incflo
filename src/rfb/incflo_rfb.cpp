#include <incflo.H>

using namespace amrex;

///////////////////////////////////////////////////////////////////////////
//
// Redox flow battery geometry
//
///////////////////////////////////////////////////////////////////////////

// This should be implemented in prob/prob_init_fluid_usr.cpp
// But moved here for convenience

void incflo::init_rfb_geometry(amrex::Box const &vbx, amrex::Box const &gbx,
                               amrex::Array4<amrex::Real> const &vel,
                               amrex::Array4<amrex::Real> const &density,
                               amrex::Array4<int> const &cell_type,
                               amrex::Box const &domain,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::init_rfb_geometry()");
#if (AMREX_SPACEDIM == 2)
    amrex::Abort("init_rfb_geometry not implemented for 2D");
#elif (AMREX_SPACEDIM == 3)

    auto const num_fibers = m_rfb_num_fibers;

    ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    // Compute cell center coordinates
                    Real x = problo[0] + (i + 0.5) * dx[0];
                    Real y = problo[1] + (j + 0.5) * dx[1];
                    Real z = problo[2] + (k + 0.5) * dx[2];

                    // Create references
                    Real &velx = vel(i, j, k, 0);
                    Real &vely = vel(i, j, k, 1);
                    Real &velz = vel(i, j, k, 2);
                    int &cell_type_ijk = cell_type(i, j, k);

                    // Initialize all cell_type to 0 (fluid)
                    cell_type_ijk = 0;

                    // Loop over all fibers
                    for (int n = 0; n < num_fibers; ++n)
                    {
                        // Filament diameter
                        Real d_filament = m_rfb_fibers[n][0];
                        // Filament length
                        Real l_filament = m_rfb_fibers[n][1];
                        // Filament radius
                        Real r_filament = d_filament / 2.0;
                        // Half filament length
                        Real half_l_filament = l_filament / 2.0;

                        // Get fiber orientation
                        Real ox = m_rfb_fibers[n][2];
                        Real oy = m_rfb_fibers[n][3];
                        Real oz = m_rfb_fibers[n][4];

                        // Get fiber center
                        Real cx = m_rfb_fibers[n][5];
                        Real cy = m_rfb_fibers[n][6];
                        Real cz = m_rfb_fibers[n][7];

                        // Solid geometry: spherocylinder
                        Real rx = x - cx;
                        Real ry = y - cy;
                        Real rz = z - cz;

                        Real rdot_axis = rx * ox + ry * oy + rz * oz;
                        Real rnorm2 = rx * rx + ry * ry + rz * rz;
                        Real rperp2 = rnorm2 - rdot_axis * rdot_axis;

                        Real cap1x = rx + half_l_filament * ox;
                        Real cap1y = ry + half_l_filament * oy;
                        Real cap1z = rz + half_l_filament * oz;
                        bool in_cap_1 = (cap1x * cap1x + cap1y * cap1y + cap1z * cap1z <= r_filament * r_filament);

                        Real cap2x = rx - half_l_filament * ox;
                        Real cap2y = ry - half_l_filament * oy;
                        Real cap2z = rz - half_l_filament * oz;
                        bool in_cap_2 = (cap2x * cap2x + cap2y * cap2y + cap2z * cap2z <= r_filament * r_filament);

                        bool in_cylinder = (amrex::Math::abs(rdot_axis) <= half_l_filament) && (rperp2 <= r_filament * r_filament);

                        if (in_cap_1 || in_cap_2 || in_cylinder)
                        {
                            cell_type_ijk = 1; // solid cell
                            velx = 0.0;
                            vely = 0.0;
                            velz = 0.0;
                        }
                    }
                });
#endif
}

void incflo::set_rfb_velocity()
{
    BL_PROFILE("incflo::set_rfb_velocity()");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("set_rfb_velocity not implemented for 2D");

#elif (AMREX_SPACEDIM == 3)

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto &ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.cell_type, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            Array4<Real> const &vel = ld.velocity.array(mfi);
            Array4<int const> const &cell_type = ld.cell_type.const_array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                // Cell_type: solid (filament)
                if (cell_type(i,j,k) == 1)
                {
                    vel(i,j,k,0) = 0.0;
                    vel(i,j,k,1) = 0.0;
                    vel(i,j,k,2) = 0.0;
                } });
        }
    }
#endif
}

void incflo::rfb_mark_fiber_cells()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto &ld = *m_leveldata[lev];
        Box const& domain = geom[lev].Domain();
        auto const& dx = geom[lev].CellSizeArray();
        auto const& problo = geom[lev].ProbLoArray();
        auto const& probhi = geom[lev].ProbHiArray();
        for (MFIter mfi(ld.cell_type, false); mfi.isValid(); ++mfi)
        {
            Box const& vbx = mfi.validbox();
            Box const& gbx = amrex::grow(vbx, ld.cell_type.nGrow());
            init_rfb_geometry(vbx, gbx, ld.velocity.array(mfi), ld.density.array(mfi),
                              ld.cell_type.array(mfi), domain, dx, problo, probhi);
        }
    }
}

amrex::iMultiFab incflo::rfb_cell_type_with_ghost(int lev) const
{
    auto const& ct = m_leveldata[lev]->cell_type;
    iMultiFab out(ct.boxArray(), ct.DistributionMap(), 1, 1);
    out.setVal(0);
    iMultiFab::Copy(out, ct, 0, 0, 1, 0);
    out.FillBoundary(geom[lev].periodicity());
    return out;
}

void incflo::rfb_mask_face_coeffs(int lev, amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& beta) const
{
    iMultiFab ct = rfb_cell_type_with_ghost(lev);
    Real const eps = m_rfb_mask_eps;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        IntVect const e = IntVect::TheDimensionVector(d);
        for (MFIter mfi(beta[d], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& fbx = mfi.tilebox();
            Array4<Real> const& b = beta[d].array(mfi);
            Array4<int const> const& c = ct.const_array(mfi);
            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (c(i,j,k) == 1 || c(i-e[0],j-e[1],k-e[2]) == 1) { b(i,j,k) *= eps; }
            });
        }
    }
}

void incflo::rfb_zero_face_velocity(int lev, amrex::Array<amrex::MultiFab*,AMREX_SPACEDIM> const& umac) const
{
    iMultiFab ct = rfb_cell_type_with_ghost(lev);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        IntVect const e = IntVect::TheDimensionVector(d);
        for (MFIter mfi(*umac[d], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& fbx = mfi.tilebox();
            Array4<Real> const& u = umac[d]->array(mfi);
            Array4<int const> const& c = ct.const_array(mfi);
            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (c(i,j,k) == 1 || c(i-e[0],j-e[1],k-e[2]) == 1) { u(i,j,k) = 0.0; }
            });
        }
    }
}
