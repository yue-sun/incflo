#include <incflo.H>

using namespace amrex;

#ifdef INCFLO_SIM_CRYO

void incflo::cryo_update (int i, int j, int k,
                            Real x, Real y, Real z,
                            Real &velx, Real &vely, Real &velz,
                            Real &cell_type_ijk,
                            Real time,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi)
{
    BL_PROFILE("incflo::cryo_update");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_update: not implemented in 2D");

// TODO: think, is there need to add the SIM_CRYO guard here?
#elif (AMREX_SPACEDIM == 3)
    
    if (m_cryo_geometry == -1) {
        // -1: debug sphere, static
        Real Rdebug = 2.5;
        // TODO: this should be in a separate file, velz_plunge
        Real velz_plunge = Real(-1.0); 
        Real zoff = 2.0*Rdebug - velz_plunge*time;
        Real geom_sphere = x*x + y*y + (z+zoff)*(z+zoff);
        if (geom_sphere < Rdebug*Rdebug) {
            cell_type_ijk = Real(-6.0);
            velx = Real(0.0);
            vely = Real(0.0);
            velz = velz_plunge;
        } else {
            cell_type_ijk = Real(-1.0);
        }
    }
#endif

}

#endif