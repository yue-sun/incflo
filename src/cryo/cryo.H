
#ifndef CRYO_
#define CRYO_

// This header is included in incflo.H.
// These functions are members of incflo class.

///////////////////////////////////////////////////////////////////////////
//
// Set geometry
//
///////////////////////////////////////////////////////////////////////////

// Debug sphere
amrex::Real rdebug = amrex::Real(0.5); // debug sphere radius: 0.5mm

// The disk-family geometries (gold EM grid / sapphire / diamond: radius,
// thickness, mesh dimensions) live in the cryo_grid::grids table inside
// cryo_grid.H — the single source of truth shared with cryo_sample.cpp.

// Sample
amrex::Real wsam = amrex::Real(0.01); // sample width: 10um


// Wiper/obstacle primitives. The POD definition lives in cryo_stamp.H so the
// per-cell stamper can be a device function; m_cryo_solids_d is the device
// mirror uploaded once by cryo_read_solids.
int m_cryo_n_solids = 0;
amrex::Vector<cryo_stamp::Solid> m_cryo_solids;
amrex::Gpu::DeviceVector<cryo_stamp::Solid> m_cryo_solids_d;

// Discrete biological samples (see cryo_sample.{H,cpp})
#include <cryo_sample.H>

void cryo_read_solids();

void cryo_plunge_state(amrex::Real time,
                       amrex::Real &velz_plunge,
                       amrex::Real &plunge_disp) const;

///////////////////////////////////////////////////////////////////////////
//
// Helper functions
//
///////////////////////////////////////////////////////////////////////////

void cryo_init();

void cryo_update(amrex::Real time = amrex::Real(-1.0));

// Host-side packing for the device stampers in cryo_stamp.H: resolve the
// geometry preset (disk-table row, its z offset) and the time-only kinematics
// once per update, so the per-cell path touches no incflo member.
cryo_stamp::DiskParams cryo_disk_params(amrex::Real velz_plunge,
                                        amrex::Real plunge_disp,
                                        cryo_tc::ThermocoupleMotion const &motion) const;

#endif
