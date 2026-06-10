
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
// Additive geometry primitives ("wiper" solids), stamped after the
// m_cryo_geometry presets. Defined in the co-moving frame
// (x, y, z' = z - plunge_disp). See inputs keys incflo.cryo_solid_*.
//
///////////////////////////////////////////////////////////////////////////

struct CryoSolid
{
    int shape = 0;                 // 0 = bar (box), 1 = vane (box rotated about y), 2 = serration carve
    int material = -6;             // cell_type stamped inside (default gold); ignored by serration
    amrex::Real cx = 0., cy = 0., cz = 0.;   // box center (bar/vane), co-moving frame
    amrex::Real hx = 0., hy = 0., hz = 0.;   // box half-extents
    amrex::Real angle = 0.;        // vane pitch about y [radians], applied in the x-z' plane
    int osc_axis = -1;             // -1 = rigid with plunger, 0 = oscillate along x, 2 = along z
    amrex::Real osc_amp = 0.;      // oscillation amplitude [mm]
    amrex::Real osc_freq = 0.;     // oscillation frequency [cycles per time unit]
    amrex::Real osc_phase = 0.;    // phase [radians]
    // serration (shape == 2): carve teeth from the disk's leading (bottom) arc
    amrex::Real serr_R = amrex::Real(1.5);     // disk radius
    amrex::Real serr_zc = amrex::Real(1.5);    // disk center z' (co-moving)
    amrex::Real serr_depth = 0.;               // radial tooth depth
    int serr_nteeth = 0;                       // teeth count over the full circle
    amrex::Real serr_duty = amrex::Real(0.5);  // tooth fraction of one pitch (rest is notch)
};

int m_cryo_n_solids = 0;
amrex::Vector<CryoSolid> m_cryo_solids;

void cryo_read_solids();

void cryo_plunge_state(amrex::Real time,
                       amrex::Real &velz_plunge,
                       amrex::Real &plunge_disp) const;

void cryo_apply_solids(amrex::Real x, amrex::Real y, amrex::Real z,
                       amrex::Real &velx, amrex::Real &vely, amrex::Real &velz,
                       int &cell_type_ijk, amrex::Real &rho_ijk,
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
