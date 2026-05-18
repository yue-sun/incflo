#include <incflo.H>

using namespace amrex;

// void 
void incflo::init_cryo_plunging (amrex::Box const& vbx, amrex::Box const& /*gbx*/,
								 amrex::Array4<amrex::Real> const& vel,
								 amrex::Array4<amrex::Real> const& density,
								 amrex::Array4<amrex::Real> const& cell_type,
								 amrex::Array4<amrex::Real> const& tracer,
								 amrex::Array4<amrex::Real> const& temperature,
								 amrex::Box const& /*domain*/,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi)
{
	return;
}