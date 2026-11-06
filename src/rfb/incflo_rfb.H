#ifndef INCFLO_RFB_
#define INCFLO_RFB_

// This header is included by incflo.H.  These functions are members of incflo class.

void init_rfb_geometry (amrex::Box const& vbx, amrex::Box const& gbx,
                        amrex::Array4<amrex::Real> const& vel,
                        amrex::Array4<amrex::Real> const& density,
                        amrex::Array4<int> const& cell_type,
                        amrex::Box const& domain,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx, 
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo, 
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi);

void set_rfb_velocity ();
// Mark the fiber cells (cell_type) of every level again, e.g. after a restart: checkpoints do not
// store cell_type. Also zeroes the velocity in the fiber cells.
void rfb_mark_fiber_cells ();
// cell_type of level lev with one ghost cell: neighbours across box and periodic boundaries are
// filled, cells outside the domain count as fluid.
amrex::iMultiFab rfb_cell_type_with_ghost (int lev) const;
// Masked MAC projection: scale the face coefficients that touch a fiber cell by m_rfb_mask_eps.
void rfb_mask_face_coeffs (int lev, amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& beta) const;
// Masked MAC projection: zero the face velocities that touch a fiber cell.
void rfb_zero_face_velocity (int lev, amrex::Array<amrex::MultiFab*,AMREX_SPACEDIM> const& umac) const;

#endif
