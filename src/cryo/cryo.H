
#ifndef CRYO_
#define CRYO_

// This header is included in incflo.H.
// These functions are members of incflo class.

///////////////////////////////////////////////////////////////////////////
//
// Set geometry
//
///////////////////////////////////////////////////////////////////////////

// TODO: see if we need cryo_geometries.H for more complex geometry definitions; if so, add the geometry-specific
// Debug sphere
amrex::Real rdebug = amrex::Real(0.5); // debug sphere radius: 0.5mm

// EM grid (gold)
amrex::Real Remg = amrex::Real(1.5);         // EM grid outer radius: 1.5mm
amrex::Real wemg = amrex::Real(0.025 / 2);   // EM grid width: 25um
amrex::Real remg = Remg - amrex::Real(0.24); // EM grid ring width: 240um

// Sapphire disk
amrex::Real Rsap = amrex::Real(1.5);      // sapphire disk radius: 1.5mm
amrex::Real wsap = amrex::Real(0.16 / 2); // sapphire disk width: 160um

// Sample
amrex::Real wsam = amrex::Real(0.01); // sample width: 10um


///////////////////////////////////////////////////////////////////////////
//
// Helper functions
//
///////////////////////////////////////////////////////////////////////////

void cryo_init();

void cryo_update(amrex::Real time = amrex::Real(-1.0));

void cryo_set_geom_velocity(int i, int j, int k,
                            amrex::Real x, amrex::Real y, amrex::Real z,
                            amrex::Real &velx, amrex::Real &vely, amrex::Real &velz,
                            int &cell_type_ijk, amrex::Real &rho_ijk,
                            amrex::Real time,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi);

void cryo_set_temp_top_bc(amrex::Real z, amrex::Real &temperature_ijk,
                          int cell_type_ijk,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi);

void cryo_set_thermal(int i, int j, int k,
                      amrex::Real x, amrex::Real y, amrex::Real z,
                      int cell_type_ijk, amrex::Real time,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi);

void cryo_load_emgrid();

void cryo_load_disk();

void cryo_load_sample();

#endif
