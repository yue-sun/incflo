#include <incflo.H>

using namespace amrex;

///////////////////////////////////////////////////////////////////////////
//
// Redox flow battery geometry
//
///////////////////////////////////////////////////////////////////////////

// TODO: Have classes to determine which RFB geometry to use

// 1. Benchmark: Minecraft vs cutcell simulation
Real d_filament = Real(35e-6);           // filament diameter
Real l_filament = Real(200e-6);          // filament length
Real r_filament = d_filament / 2.0;      // filament radius
Real half_l_filament = l_filament / 2.0; // half filament length

///////////////////////////////////////////////////////////////////////////
//
// Helper functions
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
    auto const fiber_center = m_rfb_fiber_centers;
    auto const fiber_dir = m_rfb_fiber_dirs;

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
                        Real xc = fiber_center[AMREX_SPACEDIM * n + 0];
                        Real yc = fiber_center[AMREX_SPACEDIM * n + 1];
                        Real zc = fiber_center[AMREX_SPACEDIM * n + 2];

                        Real ax = fiber_dir[AMREX_SPACEDIM * n + 0];
                        Real ay = fiber_dir[AMREX_SPACEDIM * n + 1];
                        Real az = fiber_dir[AMREX_SPACEDIM * n + 2];

                        // Solid geometry: spherocylinder
                        Real rx = x - xc;
                        Real ry = y - yc;
                        Real rz = z - zc;

                        Real rdot_axis = rx * ax + ry * ay + rz * az;
                        Real rnorm2 = rx * rx + ry * ry + rz * rz;
                        Real rperp2 = rnorm2 - rdot_axis * rdot_axis;

                        Real cap1x = rx + half_l_filament * ax;
                        Real cap1y = ry + half_l_filament * ay;
                        Real cap1z = rz + half_l_filament * az;
                        bool in_cap_1 = (cap1x * cap1x + cap1y * cap1y + cap1z * cap1z <= r_filament * r_filament);

                        Real cap2x = rx - half_l_filament * ax;
                        Real cap2y = ry - half_l_filament * ay;
                        Real cap2z = rz - half_l_filament * az;
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