#include <incflo.H>

using namespace amrex;

#ifdef INCFLO_SIM_CRYO

void incflo::cryo_update()
{
    BL_PROFILE("incflo::cryo_update");

    if (!m_sim_cryo)
    {
        return;
    }

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_update: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto &ld = *m_leveldata[lev];
        //  Box const& domain = geom[lev].Domain();
        auto const &dx = geom[lev].CellSizeArray();
        auto const &problo = geom[lev].ProbLoArray();
        auto const &probhi = geom[lev].ProbHiArray();

        for (MFIter mfi(ld.density, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            Array4<Real> const &vel = ld.velocity.array(mfi);
            Array4<int> const &cell_type = ld.cell_type.array(mfi);
            Array4<Real> const &temperature = ld.temperature.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            Real x = (i + 0.5) * dx[0] - 0.5 * (probhi[0] - problo[0]);
                            Real y = (j + 0.5) * dx[1] - 0.5 * (probhi[1] - problo[1]);
                            Real z = (k + 0.5) * dx[2] - (probhi[2] - problo[2]);

                            int &cell_type_ijk = cell_type(i, j, k);
                            Real &velx = vel(i, j, k, 0);
                            Real &vely = vel(i, j, k, 1);
                            Real &velz = vel(i, j, k, 2);
                            Real &temperature_ijk = temperature(i, j, k);

                            // Set geometry and velocity
                            cryo_set_geom_velocity(i, j, k, x, y, z,
                                                   velx, vely, velz,
                                                   cell_type_ijk,
                                                   m_cur_time, dx, problo, probhi);
                            // Set thermal properties and top temperature B.C.
                            cryo_set_thermal(i, j, k, x, y, z,
                                             cell_type_ijk,
                                             m_cur_time, dx, problo, probhi);
                            // Set top boundary condition for temperature/heat
                            cryo_set_temp_top_bc(z, temperature_ijk, cell_type_ijk, dx, probhi);
                        });
        }
    }
#endif
}

void incflo::cryo_set_geom_velocity(int i, int j, int k,
                                    Real x, Real y, Real z,
                                    Real &velx, Real &vely, Real &velz,
                                    int &cell_type_ijk,
                                    Real time,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_geom_velocity");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_update: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)

    // ***************************************************************
    // Plunging protocols
    // ***************************************************************

    // Default plunging protocol
    Real velz_plunge = Real(-1.0);
    Real plunge_disp = velz_plunge * time;

    // Load prescribed plunging protocol if provided
    if (!m_cryo_plunge_vel.empty() &&
        m_cryo_plunge_vel.size() == m_cryo_plunge_time.size())
    {
        int const nintervals = static_cast<int>(m_cryo_plunge_vel.size());
        velz_plunge = m_cryo_plunge_vel[0];
        plunge_disp = Real(0.0);

        if (time > m_cryo_plunge_time[0])
        {
            for (int n = 0; n < nintervals - 1; ++n)
            {
                Real const t0 = m_cryo_plunge_time[n];
                Real const t1 = m_cryo_plunge_time[n + 1];

                if (time < t1)
                {
                    velz_plunge = m_cryo_plunge_vel[n];
                    plunge_disp += m_cryo_plunge_vel[n] * (time - t0);
                    break;
                }

                plunge_disp += m_cryo_plunge_vel[n] * (t1 - t0);
                velz_plunge = m_cryo_plunge_vel[n + 1];
            }

            if (time >= m_cryo_plunge_time[nintervals - 1])
            {
                plunge_disp += m_cryo_plunge_vel[nintervals - 1] *
                               (time - m_cryo_plunge_time[nintervals - 1]);
                velz_plunge = m_cryo_plunge_vel[nintervals - 1];
            }
        }
    }

    // ***************************************************************
    // Set geometry and velocity based on the plunging protocol
    // ***************************************************************

    if (m_cryo_geometry == -1)
    {
        // -1: debug sphere, static
        Real Rdebug = 2.5;
        Real zoff = 2.0 * Rdebug - plunge_disp;
        Real geom_sphere = x * x + y * y + (z + zoff) * (z + zoff);
        if (geom_sphere < Rdebug * Rdebug)
        {
            cell_type_ijk = -6;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = velz_plunge;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }
    else if (m_cryo_geometry == 1)
    {
        // 1: sapphire disk
        Real R_sap_disk = Real(1.5);      // sapphire disk radius: 1.5mm
        Real w_sap_disk = Real(0.16 / 2); // sapphire disk width: 160um

        Real zoff = R_sap_disk + plunge_disp;
        Real geom_sap_disk = x * x + (z - zoff) * (z - zoff);
        Real geom_sap_disk_thickness = amrex::Math::abs(y);
        // Disk geometry
        if (geom_sap_disk < R_sap_disk * R_sap_disk &&
            geom_sap_disk_thickness < w_sap_disk)
        {
            cell_type_ijk = -4;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = velz_plunge;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }
#endif
}

void incflo::cryo_set_thermal(int i, int j, int k,
                              Real x, Real y, Real z,
                              int cell_type_ijk,
                              Real time,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_thermal");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_set_thermal: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)
    //
    return;
#endif
}

void incflo::cryo_set_temp_top_bc(Real z, Real &temperature_ijk, int cell_type_ijk,
                                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_temp_top_bc");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_set_temp_top_bc: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)
    // TODO: add a check on the m_cryo_geometry type
    if (z >= probhi[2] - dx[2] && cell_type_ijk != -1)
    {
        // Set top temperature boundary condition for solid cells
        temperature_ijk = m_cryo_temp_entry;
    }
#endif
}

#endif