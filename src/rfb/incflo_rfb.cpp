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