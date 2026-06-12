#include <incflo.H>

using namespace amrex;

void incflo::init_cryo_plunging (amrex::Box const& vbx, amrex::Box const& /*gbx*/,
								 amrex::Array4<amrex::Real> const& vel,
								 amrex::Array4<amrex::Real> const& density,
                                 amrex::Array4<int> const& cell_type,
								 amrex::Array4<amrex::Real> const& tracer,
								 amrex::Array4<amrex::Real> const& temperature,
								 amrex::Box const& /*domain*/,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi)
{
#if (AMREX_SPACEDIM == 2)
    amrex::Abort("init_cryo_plugning: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = (i+0.5)*dx[0] - 0.5*(probhi[0] - problo[0]);
        Real y = (j+0.5)*dx[1] - 0.5*(probhi[1] - problo[1]);
        Real z = (k+0.5)*dx[2] -     (probhi[2] - problo[2]);

        Real &velx = vel(i,j,k,0);
        Real &vely = vel(i,j,k,1);
        Real &velz = vel(i,j,k,2);
        int &cell_type_ijk = cell_type(i,j,k);
        Real &temperature_ijk = temperature(i,j,k);

        // Initialize density, velocity, and cell_type for each cell
        density(i,j,k) = Real(1.0);
        velx = vely = velz = Real(0.0);
        cell_type_ijk = -1;

        // Update velocity and cell_type for each cell
        // based on the prescribed velocity of the plunging protocol
#ifdef INCFLO_SIM_CRYO
        cryo_set_geom_velocity(i, j, k, x, y, z,
                            velx, vely, velz, cell_type_ijk,
                            m_cur_time, dx, problo, probhi);
        cryo_set_thermal(i, j, k, x, y, z, cell_type_ijk, m_cur_time, dx, problo, probhi);

        // Set initial temperature
        if (cell_type_ijk == -1) {
            // -1: liquid ethane (fluid)
            temperature_ijk = m_cryo_temp_eth;
        } else {
            // All other cell types for solids
            temperature_ijk = m_cryo_temp_entry;
        }
#endif

        // Update thermal properties for each cell
        // -1: liquid ethane (fluid)
        // -2: thermocouple (solid), 
        // -3: EM grid (solid), 
        // -4: sapphire grid (solid), 
        // -5: diamond grid (solid),
        // -6: debug sphere
        // TODO: if (cell_type_ijk == -1.0) {
        //     temperature_ijk = ...;
        // }

    });
#endif
}