
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
// thickness, mesh dimensions) live in the cryo_geom::disks table in
// cryo_geometries.H — the single source of truth shared with cryo_sample.cpp.

// Sample
amrex::Real wsam = amrex::Real(0.01); // sample width: 10um


///////////////////////////////////////////////////////////////////////////
//
// Additive geometry primitives ("wiper" solids), stamped after the
// m_cryo_geometry presets. Defined in the co-moving frame
// (x, y, z' = z - plunge_disp). See inputs keys incflo.cryo_solid_*.
//
///////////////////////////////////////////////////////////////////////////

struct CryoSolid
{
    int shape = 0;                 // 0 = bar (box), 1 = vane (box rotated about y), 2 = serration carve, 3 = ring (annulus about y axis), 4 = gear_rim
    int attach = 0;                // 0 = plunger (co-moving frame z'), 1 = lab-fixed (frame z, stamped velocity 0)
    int material = -7;             // cell_type stamped inside (default PTFE wiper); ignored by serration
    amrex::Real cx = 0., cy = 0., cz = 0.;   // box center (bar/vane), co-moving frame
    amrex::Real hx = 0., hy = 0., hz = 0.;   // box half-extents
    amrex::Real angle = 0.;        // vane pitch about y [radians], applied in the x-z' plane
    int osc_axis = -1;             // -1 = rigid with plunger, 0 = oscillate along x, 2 = along z
    amrex::Real osc_amp = 0.;      // oscillation amplitude [mm]
    amrex::Real osc_freq = 0.;     // oscillation frequency [cycles per time unit]
    amrex::Real osc_phase = 0.;    // phase [radians]
    amrex::Real ring_r0 = 0., ring_r1 = 0.;    // ring inner/outer radius (shape == 3; uses hy for half-thickness in y)
    // serration (shape == 2): carve teeth from the disk's leading (bottom) arc
    amrex::Real serr_R = amrex::Real(1.5);     // disk radius
    amrex::Real serr_zc = amrex::Real(1.5);    // disk center z' (co-moving)
    amrex::Real serr_depth = 0.;               // radial tooth depth
    int serr_nteeth = 0;                       // teeth count over the full circle
    amrex::Real serr_duty = amrex::Real(0.5);  // tooth fraction of one pitch (rest is notch)
    // gear_rim (shape == 4): annular rim at the disk edge with y-extruding teeth
    amrex::Real rim_r_inner = 0.;              // inner radius of the rim annulus
    amrex::Real rim_r_outer = 0.;              // outer radius (typically = disk radius)
    amrex::Real rim_zc = amrex::Real(1.5);     // disk center z' (co-moving)
    amrex::Real rim_base_hy = 0.;             // y half-thickness of the base ring (typically = disk half-width)
    amrex::Real rim_tooth_h = 0.;             // additional y extrusion of each tooth beyond ±rim_base_hy
    int rim_nteeth = 0;                        // number of teeth around the full ring
    amrex::Real rim_duty = amrex::Real(0.5);   // tooth angular fraction per pitch (0..1)
};

int m_cryo_n_solids = 0;
amrex::Vector<CryoSolid> m_cryo_solids;

// Discrete biological samples (see cryo_sample.{H,cpp})
#include <cryo_sample.H>

void cryo_read_solids();

void cryo_plunge_state(amrex::Real time,
                       amrex::Real &velz_plunge,
                       amrex::Real &plunge_disp) const;

void cryo_apply_solids(amrex::Real x, amrex::Real y, amrex::Real z,
                       amrex::Real &velx, amrex::Real &vely, amrex::Real &velz,
                       int &cell_type_ijk,
                       amrex::Real time,
                       amrex::Real velz_plunge, amrex::Real plunge_disp) const;

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
                            int &cell_type_ijk,
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

#endif
