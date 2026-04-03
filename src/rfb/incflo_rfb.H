#ifndef INCFLO_RFB_
#define INCFLO_RFB_

// This header is included by incflo.H.  These functions are members of incflo class.

void init_rfb_geometry (amrex::Box const& vbx, amrex::Box const& gbx,
                        amrex::Array4<amrex::Real> const& vel,
                        amrex::Array4<amrex::Real> const& density,
                        amrex::Array4<amrex::Real> const& cell_type,
                        amrex::Box const& domain,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx, 
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo, 
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi);

void set_rfb_velocity ();

#endif
