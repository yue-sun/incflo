#ifndef INCFLO_CRYO_STAMP_H_
#define INCFLO_CRYO_STAMP_H_

#include <AMReX.H>            // amrex::ignore_unused
#include <AMReX_REAL.H>
#include <AMReX_INT.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Array.H>
#include <AMReX_Math.H>
#include <AMReX_SPACE.H>

#include <cmath>

#include <cryo_grid.H>
#include <cryo_tc_experiment.H>

// ===========================================================================
// Device-callable per-cell stamping of the cryo geometry.
//
// Everything here is a free AMREX_GPU_HOST_DEVICE function over POD inputs.
// The incflo members and host containers they used to read are packed ONCE per
// cryo_update into the small structs below, because
//   * a device lambda may capture `this` but must not dereference it (host
//     pointer on the device), and
//   * AMReX is built without relocatable device code, so a __device__ function
//     must be visible in the translation unit that launches the kernel --
//     hence a header rather than cryo.cpp.
//
// The geometry data itself lives in cryo_grid.H (disk table + apply_disk); the
// thermocouple motion spline stays host-only and is evaluated once per update
// (it is time-only) and passed in by value.
// ===========================================================================

namespace cryo_stamp
{

    ///////////////////////////////////////////////////////////////////////////
    //
    // Additive geometry primitives ("wiper" solids), stamped after the
    // cryo_geometry presets. Defined in the co-moving frame
    // (x, y, z' = z - plunge_disp). See inputs keys incflo.cryo_solid_*.
    //
    // POD by construction so the whole array can be uploaded to the device
    // once at read time (incflo::m_cryo_solids_d).
    //
    ///////////////////////////////////////////////////////////////////////////

    struct Solid
    {
        int shape = 0;                 // 0 = bar (box), 1 = vane (box rotated about y), 2 = serration carve, 3 = ring (annulus about y axis), 4 = gear_rim
        int attach = 0;                // 0 = plunger (co-moving frame z'), 1 = lab-fixed (frame z, stamped velocity 0)
        int material = -7;             // cell_type stamped inside (default PTFE wiper); ignored by serration
        amrex::Real cx = 0., cy = 0., cz = 0.;   // box center (bar/vane), co-moving frame
        amrex::Real hx = 0., hy = 0., hz = 0.;   // box half-extents
        amrex::Real angle = 0.;        // vane pitch about y [radians], applied in the x-z' plane
        amrex::Real cos_angle = 1.;    // cos/sin of `angle`, cached at read time (they are
        amrex::Real sin_angle = 0.;    // parse-time constants; recomputing per cell is pure waste)
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
        amrex::Real rim_base_hy = 0.;              // y half-thickness of the base ring (typically = disk half-width)
        amrex::Real rim_tooth_h = 0.;              // additional y extrusion of each tooth beyond ±rim_base_hy
        int rim_nteeth = 0;                        // number of teeth around the full ring
        amrex::Real rim_duty = amrex::Real(0.5);   // tooth angular fraction per pitch (0..1)
    };

    ///////////////////////////////////////////////////////////////////////////
    //
    // Discrete biological samples: device views of the host-side VoxelShape /
    // CryoCell records built by incflo::cryo_read_samples. The occupancy masks
    // of all shapes are concatenated into one buffer, so a shape is (dims,
    // offset) and the whole set is three flat arrays.
    //
    ///////////////////////////////////////////////////////////////////////////

    struct VoxelView
    {
        int nx = 0, ny = 0, nz = 0;
        amrex::Real voxel_size_mm = amrex::Real(0.0);
        amrex::Long occ_offset = 0;             // start of this shape inside the shared mask buffer
    };

    struct CellView
    {
        int shape_idx = 0;                      // index into the VoxelView array
        amrex::Real cx = amrex::Real(0.0);      // disk-plane center offset x [mm]
        amrex::Real cz = amrex::Real(0.0);      // disk-plane center offset z [mm]
        amrex::Real cos_neg_angle = amrex::Real(1.0);   // cos(-angle) / sin(-angle), cached at read
        amrex::Real sin_neg_angle = amrex::Real(0.0);   // time (the un-rotation is by -angle)
        int flip = 0;                           // mirror local x before rotation
        amrex::Real scale = amrex::Real(1.0);   // multiplies voxel_size_mm
        int label = 1;                          // cell_type (1..n)
    };

    // Everything the per-cell sample stamp needs: the three device arrays plus
    // the host disk geometry the samples sit on (face height, radius, centre z).
    struct SampleData
    {
        CellView  const* cells = nullptr;
        VoxelView const* shapes = nullptr;
        char      const* occ = nullptr;
        int n_cells = 0;
        bool host_valid = false;                // geometry hosts samples at all
        amrex::Real face_y = amrex::Real(0.0);
        amrex::Real radius = amrex::Real(0.0);
        amrex::Real zoff = amrex::Real(0.0);
    };

    ///////////////////////////////////////////////////////////////////////////
    //
    // Geometry-preset parameters, resolved once per cryo_update: the disk-table
    // lookup, its z offset for this time, and the time-only kinematics.
    //
    ///////////////////////////////////////////////////////////////////////////

    struct DiskParams
    {
        int geometry = 0;                       // incflo.cryo_geometry
        bool is_disk = false;                   // geometry is in the cryo_grid table
        cryo_grid::GridGeom grid{};             // the table row (valid when is_disk)
        amrex::Real zoff = amrex::Real(0.0);    // disk centre along z = init_z + plunge_disp
        int sample_layer = 0;                   // continuous +y-face sample layer on/off
        amrex::Real sample_layer_thickness = amrex::Real(0.0);
        amrex::Real velz_plunge = amrex::Real(0.0);
        amrex::Real plunge_disp = amrex::Real(0.0);
        cryo_tc::ThermocoupleMotion motion{};   // evaluated once per update (time-only)
    };

    ///////////////////////////////////////////////////////////////////////////
    //
    // Per-cell stampers
    //
    ///////////////////////////////////////////////////////////////////////////

    // Geometry presets: disk family (gold EM grid / sapphire / diamond), debug
    // sphere, thermocouple beads. Sets cell_type and the rigid-body velocity.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_geom_velocity (amrex::Real x, amrex::Real y, amrex::Real z,
                            amrex::Real& velx, amrex::Real& vely, amrex::Real& velz,
                            int& cell_type_ijk, DiskParams const& p) noexcept
    {
#if (AMREX_SPACEDIM == 3)
        if (p.is_disk)
        {
            // Disk family (gold EM grid / sapphire / diamond): a thin disk in the
            // x-z plane, normal along y, plunged edge-on. The disk body and its
            // optional +y-face sample layer co-move, so share the plunge velocity.
            // See cryo_grid.H for the per-material table.
            if (!cryo_grid::apply_disk(p.grid, x, y, z, p.zoff,
                                       p.sample_layer, p.sample_layer_thickness,
                                       cell_type_ijk, velx, vely, velz, p.velz_plunge))
            {
                cell_type_ijk = -1;
            }
        }
        else if (p.geometry == -1)
        {
            // -1: debug sphere, static
            amrex::Real Rdebug = 0.2;
            amrex::Real zoff = -2. * Rdebug + p.plunge_disp;
            amrex::Real geom_sphere = x * x + y * y + (z - zoff) * (z - zoff);
            if (geom_sphere < Rdebug * Rdebug)
            {
                cell_type_ijk = -6;
                velx = amrex::Real(0.0);
                vely = amrex::Real(0.0);
                velz = p.velz_plunge;
            }
            else
            {
                cell_type_ijk = -1;
            }
        }
        else if (p.geometry == 10 || p.geometry == 11 || p.geometry == 12)
        {
            // Thermocouple probe, modelled as a sphere centred on the plunge
            // path, moving with the measured linear-motor kinematics. One table
            // of bead radii instead of three copies of the same block:
            //   10:  60 um bead diameter with  25 um wire
            //   11:  85 um bead diameter with  25 um wire
            //   12: 420 um bead diameter with 127 um wire
            // The existing cryo setup uses cell_type -2 for thermocouple material.
            amrex::Real const rtc1 = (p.geometry == 10) ? amrex::Real(0.03)
                                   : (p.geometry == 11) ? amrex::Real(0.0425)
                                                        : amrex::Real(0.21);
            amrex::Real const zz = z - rtc1 + p.motion.depth;
            amrex::Real const geometry = x * x + y * y + zz * zz;

            if (geometry < rtc1 * rtc1)
            {
                cell_type_ijk = -2;
                velx = amrex::Real(0.0);
                vely = amrex::Real(0.0);
                velz = -p.motion.speed;
            }
            else
            {
                cell_type_ijk = -1;
            }
        }
#else
        amrex::ignore_unused(x, y, z, velx, vely, velz, cell_type_ijk, p);
#endif
    }

    // Additive wiper / obstacle primitives, stamped over the preset geometry.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void apply_solids (amrex::Real x, amrex::Real y, amrex::Real z,
                       amrex::Real& velx, amrex::Real& vely, amrex::Real& velz,
                       int& cell_type_ijk,
                       Solid const* solids, int n_solids, amrex::Real time,
                       amrex::Real velz_plunge, amrex::Real plunge_disp) noexcept
    {
        using amrex::Real;

        Real const zp = z - plunge_disp; // co-moving frame

        for (int n = 0; n < n_solids; ++n)
        {
            Solid const& s = solids[n];

            if (s.shape == 2)
            {
                // Serration: carve notches from the leading (bottom, z' < zc) arc
                // back to fluid. Velocity is left as-is, matching cells the moving
                // solid vacates.
                if (cell_type_ijk == -1) { continue; }
                Real const rx = x;
                Real const rz = zp - s.serr_zc;
                if (rz >= Real(0.0)) { continue; }            // trailing half: untouched
                Real const r2 = rx * rx + rz * rz;
                Real const Ri = s.serr_R - s.serr_depth;
                if (r2 <= Ri * Ri) { continue; }              // inside tooth root circle
                Real const theta = std::atan2(rx, -rz);       // 0 at bottom of arc
                Real frac = theta * Real(s.serr_nteeth) / Real(2.0 * M_PI);
                frac -= std::floor(frac);
                if (frac >= s.serr_duty)
                {
                    cell_type_ijk = -1;
                }
                continue;
            }

            if (s.shape == 4)
            {
                // Gear-rim: annular ring at the disk edge with teeth extruding in ±y.
                // Base ring: r in [rim_r_inner, rim_r_outer], |y| < rim_base_hy.
                // Teeth:     at tooth angular sectors, |y| < rim_base_hy + rim_tooth_h.
                Real const rx = x;
                Real const rz = zp - s.rim_zc;
                Real const r2 = rx * rx + rz * rz;
                if (r2 < s.rim_r_inner * s.rim_r_inner) { continue; }
                if (r2 > s.rim_r_outer * s.rim_r_outer) { continue; }
                Real const aby = amrex::Math::abs(y);
                if (aby >= s.rim_base_hy + s.rim_tooth_h) { continue; }
                if (aby < s.rim_base_hy)
                {
                    // inside base ring regardless of angle
                }
                else
                {
                    // in the tooth extension zone — check angular sector
                    Real const theta = std::atan2(rx, rz);   // [-pi, pi]
                    Real frac = (theta + Real(M_PI)) * Real(s.rim_nteeth) / Real(2.0 * M_PI);
                    frac -= std::floor(frac);
                    if (frac >= s.rim_duty) { continue; }    // gap sector: not a tooth
                }
                cell_type_ijk = s.material;
                velx = Real(0.0);
                vely = Real(0.0);
                velz = velz_plunge;
                continue;
            }

            // bar / vane / ring: inside-test in the (possibly oscillation-shifted,
            // possibly rotated) reference frame. attach = plunger uses the
            // co-moving frame z'; attach = lab uses the lab frame z with a zero
            // base velocity (the disk sweeps past the obstacle).
            Real const zf = (s.attach == 1) ? z : zp;
            Real const vbase = (s.attach == 1) ? Real(0.0) : velz_plunge;

            Real offx = Real(0.0), offz = Real(0.0);
            Real voscx = Real(0.0), voscz = Real(0.0);
            if (s.osc_axis >= 0 && s.osc_amp != Real(0.0) && s.osc_freq != Real(0.0))
            {
                Real const w = Real(2.0 * M_PI) * s.osc_freq;
                Real const d = s.osc_amp * std::sin(w * time + s.osc_phase);
                Real const v = s.osc_amp * w * std::cos(w * time + s.osc_phase);
                if (s.osc_axis == 0) { offx = d; voscx = v; }
                else                 { offz = d; voscz = v; }
            }

            Real px = x - (s.cx + offx);
            Real const py = y - s.cy;
            Real pz = zf - (s.cz + offz);

            bool inside = false;
            if (s.shape == 3)
            {
                // ring: annulus about the y axis (co-planar with the disk)
                Real const rr2 = px * px + pz * pz;
                inside = amrex::Math::abs(py) < s.hy &&
                         rr2 > s.ring_r0 * s.ring_r0 &&
                         rr2 < s.ring_r1 * s.ring_r1;
            }
            else
            {
                if (s.angle != Real(0.0))
                {
                    Real const c = s.cos_angle;
                    Real const sn = s.sin_angle;
                    Real const qx = c * px + sn * pz;
                    Real const qz = -sn * px + c * pz;
                    px = qx;
                    pz = qz;
                }
                inside = amrex::Math::abs(px) < s.hx &&
                         amrex::Math::abs(py) < s.hy &&
                         amrex::Math::abs(pz) < s.hz;
            }

            if (inside)
            {
                cell_type_ijk = s.material;
                velx = voscx;
                vely = Real(0.0);
                velz = vbase + voscz;
            }
        }
    }

    // Voxel occupancy of one shape inside the shared mask buffer.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool occupied (VoxelView const& s, char const* occ, int i, int j, int k) noexcept
    {
        if (i < 0 || j < 0 || k < 0 || i >= s.nx || j >= s.ny || k >= s.nz) { return false; }
        return occ[s.occ_offset + i + s.nx * (j + s.ny * k)] != 0;
    }

    // Discrete biological samples on the host disk's +y face. First match in
    // placement order wins.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void apply_samples (amrex::Real x, amrex::Real y, amrex::Real z,
                        amrex::Real& velx, amrex::Real& vely, amrex::Real& velz,
                        int& cell_type_ijk, amrex::Real velz_plunge,
                        SampleData const& sd) noexcept
    {
        using amrex::Real;

        // Off the disk footprint: nothing to host.
        Real const rz = z - sd.zoff;
        if (x * x + rz * rz > sd.radius * sd.radius) { return; }
        Real const ly = y - sd.face_y;
        if (ly < Real(0.0)) { return; }

        for (int n = 0; n < sd.n_cells; ++n)
        {
            CellView const& c = sd.cells[n];
            VoxelView const& s = sd.shapes[c.shape_idx];
            Real const v = c.scale * s.voxel_size_mm;
            if (ly > Real(s.ny) * v) { continue; }      // above this cell's height

            // local in-plane coords relative to the cell center
            Real lx = x - c.cx;
            Real lz = rz - c.cz;
            // un-rotate by -angle about +y
            Real const ca = c.cos_neg_angle, sa = c.sin_neg_angle;
            Real lxr = ca * lx - sa * lz;
            Real const lzr = sa * lx + ca * lz;
            if (c.flip) { lxr = -lxr; }

            // to voxel indices (centered in i,k; j from the face)
            int const vi = static_cast<int>(std::floor(lxr / v + Real(0.5) * s.nx));
            int const vj = static_cast<int>(std::floor(ly  / v));
            int const vk = static_cast<int>(std::floor(lzr / v + Real(0.5) * s.nz));

            if (occupied(s, sd.occ, vi, vj, vk))
            {
                cell_type_ijk = c.label;
                velx = Real(0.0);
                vely = Real(0.0);
                velz = velz_plunge;      // rigid, co-moves with the disk
                return;                  // first match (placement order) wins
            }
        }
    }

    // Per-cell thermal stamping hook. Intentionally a no-op today: cp(T)/k(T)
    // are filled from cell_type by compute_cp / compute_temperature_diff_coeff,
    // not here. Kept as the place to put a position- or time-dependent thermal
    // override if one is ever needed.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_thermal (amrex::Real x, amrex::Real y, amrex::Real z,
                      int cell_type_ijk, amrex::Real time) noexcept
    {
        amrex::ignore_unused(x, y, z, cell_type_ijk, time);
    }

    // Top-plane temperature boundary condition: hold solid cells in the top
    // layer of the domain at the entry temperature (the plunger arm above the
    // domain stays warm).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_temp_top_bc (amrex::Real z, amrex::Real& temperature_ijk, int cell_type_ijk,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
                          amrex::Real temp_entry) noexcept
    {
#if (AMREX_SPACEDIM == 3)
        // TODO: add a check on the m_cryo_geometry type
        if (z >= probhi[2] - dx[2] && cell_type_ijk != -1)
        {
            // Set top temperature boundary condition for solid cells
            temperature_ijk = temp_entry;
        }
        //  else {
        //     // Set top temperature boundary condition for fluid cells
        //     temperature_ijk = m_cryo_temp_eth;
        // }
#else
        amrex::ignore_unused(z, temperature_ijk, cell_type_ijk, dx, probhi, temp_entry);
#endif
    }

} // namespace cryo_stamp

#endif
