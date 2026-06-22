#include <incflo.H>
#include <cryo_tc_experiment.H>
#include <cryo_grid.H>

using namespace amrex;

#ifdef INCFLO_SIM_CRYO

void incflo::cryo_update(Real time)
{
    BL_PROFILE("incflo::cryo_update");

    if (!m_sim_cryo)
    {
        return;
    }

    if (time < Real(0.0))
    {
        time = m_cur_time;
    }

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_update: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)

    Real velz_plunge, plunge_disp;
    cryo_plunge_state(time, velz_plunge, plunge_disp);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto &ld = *m_leveldata[lev];
        //  Box const& domain = geom[lev].Domain();
        auto const &dx = geom[lev].CellSizeArray();
        auto const &problo = geom[lev].ProbLoArray();
        auto const &probhi = geom[lev].ProbHiArray();

        // Sample grid surface (constant across cells/levels for this geometry).
        Real sample_face_y = Real(0.0), sample_radius = Real(0.0), sample_zoff = Real(0.0);
        bool const sample_grid_ok =
            (m_cryo_n_samples > 0) &&
            cryo_sample_grid(m_cryo_geometry, plunge_disp,
                             sample_face_y, sample_radius, sample_zoff);

        for (MFIter mfi(ld.density, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            Array4<Real> const &vel = ld.velocity.array(mfi);
            Array4<int> const &cell_type = ld.cell_type.array(mfi);
            Array4<Real> const &temperature = ld.temperature.array(mfi);

            ParallelFor(bx, [=, this] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            Real x = (i + 0.5) * dx[0] - 0.5 * (probhi[0] - problo[0]);
                            Real y = (j + 0.5) * dx[1] - 0.5 * (probhi[1] - problo[1]);
                            Real z = (k + 0.5) * dx[2] - (probhi[2] - problo[2]);

                            int &cell_type_ijk = cell_type(i, j, k);
                            Real &velx = vel(i, j, k, 0);
                            Real &vely = vel(i, j, k, 1);
                            Real &velz = vel(i, j, k, 2);
                            Real &temperature_ijk = temperature(i, j, k);

                            // Set geometry and velocity
                            cryo_set_geom_velocity(i, j, k, x, y, z,
                                                   velx, vely, velz,
                                                   cell_type_ijk,
                                               time, dx, problo, probhi);
                            // Additive wiper/obstacle primitives
                            if (m_cryo_n_solids > 0)
                            {
                                cryo_apply_solids(x, y, z,
                                                  velx, vely, velz,
                                                  cell_type_ijk,
                                                  time, velz_plunge, plunge_disp);
                            }
                            // Discrete biological samples on the disk face
                            if (sample_grid_ok)
                            {
                                cryo_apply_samples(x, y, z,
                                                   velx, vely, velz,
                                                   cell_type_ijk, velz_plunge,
                                                   sample_face_y, sample_radius,
                                                   sample_zoff);
                            }
                            // Set thermal properties and top temperature B.C.
                            cryo_set_thermal(i, j, k, x, y, z,
                                             cell_type_ijk,
                                           time, dx, problo, probhi);
                            // Set top boundary condition for temperature/heat
                            cryo_set_temp_top_bc(z, temperature_ijk, cell_type_ijk, dx, probhi); });
        }
    }
#endif
}

void incflo::cryo_plunge_state(Real time, Real &velz_plunge, Real &plunge_disp) const
{
    // Default plunging protocol
    velz_plunge = Real(-1.0);
    plunge_disp = velz_plunge * time;

    // Load prescribed plunging protocol if provided
    if (!m_cryo_plunge_vel.empty() &&
        m_cryo_plunge_vel.size() == m_cryo_plunge_time.size())
    {
        int const nintervals = static_cast<int>(m_cryo_plunge_vel.size());
        velz_plunge = m_cryo_plunge_vel[0];
        plunge_disp = Real(0.0);

        if (time > m_cryo_plunge_time[0])
        {
            for (int n = 0; n < nintervals - 1; ++n)
            {
                Real const t0 = m_cryo_plunge_time[n];
                Real const t1 = m_cryo_plunge_time[n + 1];

                if (time < t1)
                {
                    velz_plunge = m_cryo_plunge_vel[n];
                    plunge_disp += m_cryo_plunge_vel[n] * (time - t0);
                    break;
                }

                plunge_disp += m_cryo_plunge_vel[n] * (t1 - t0);
                velz_plunge = m_cryo_plunge_vel[n + 1];
            }

            if (time >= m_cryo_plunge_time[nintervals - 1])
            {
                plunge_disp += m_cryo_plunge_vel[nintervals - 1] *
                               (time - m_cryo_plunge_time[nintervals - 1]);
                velz_plunge = m_cryo_plunge_vel[nintervals - 1];
            }
        }
    }
}

void incflo::cryo_read_solids()
{
    ParmParse pp("incflo");
    pp.query("cryo_n_solids", m_cryo_n_solids);
    m_cryo_solids.resize(m_cryo_n_solids);

    for (int n = 0; n < m_cryo_n_solids; ++n)
    {
        auto &s = m_cryo_solids[n];
        std::string const pre = "cryo_solid_" + std::to_string(n) + "_";

        std::string shape = "bar";
        pp.query((pre + "shape").c_str(), shape);
        if (shape == "bar") { s.shape = 0; }
        else if (shape == "vane") { s.shape = 1; }
        else if (shape == "serration") { s.shape = 2; }
        else if (shape == "ring") { s.shape = 3; }
        else if (shape == "gear_rim") { s.shape = 4; }
        else { amrex::Abort(pre + "shape must be bar, vane, serration, ring, or gear_rim"); }

        std::string attach = "plunger";
        pp.query((pre + "attach").c_str(), attach);
        if (attach == "plunger") { s.attach = 0; }
        else if (attach == "lab") { s.attach = 1; }
        else { amrex::Abort(pre + "attach must be plunger or lab"); }
        if ((s.shape == 2 || s.shape == 4) && s.attach != 0)
        {
            amrex::Abort(pre + "serration and gear_rim must be attach = plunger (they co-move with the disk)");
        }

        pp.query((pre + "material").c_str(), s.material);
        if (s.material != -2 && s.material != -3 && s.material != -4 &&
            s.material != -5 && s.material != -6 && s.material != -7)
        {
            amrex::Abort(pre + "material must be a solid cell_type (-2..-7)");
        }

        Vector<Real> v3;
        pp.queryarr((pre + "center").c_str(), v3);
        if (v3.size() == 3) { s.cx = v3[0]; s.cy = v3[1]; s.cz = v3[2]; }
        v3.clear();
        pp.queryarr((pre + "half").c_str(), v3);
        if (v3.size() == 3) { s.hx = v3[0]; s.hy = v3[1]; s.hz = v3[2]; }

        Real angle_deg = Real(0.0);
        pp.query((pre + "angle").c_str(), angle_deg);
        s.angle = angle_deg * Real(M_PI / 180.0);

        pp.query((pre + "osc_axis").c_str(), s.osc_axis);
        pp.query((pre + "osc_amp").c_str(), s.osc_amp);
        pp.query((pre + "osc_freq").c_str(), s.osc_freq);
        pp.query((pre + "osc_phase").c_str(), s.osc_phase);
        if (s.osc_axis != -1 && s.osc_axis != 0 && s.osc_axis != 2)
        {
            amrex::Abort(pre + "osc_axis must be -1 (none), 0 (x), or 2 (z)");
        }

        pp.query((pre + "serr_radius").c_str(), s.serr_R);
        pp.query((pre + "serr_zc").c_str(), s.serr_zc);
        pp.query((pre + "serr_depth").c_str(), s.serr_depth);
        pp.query((pre + "serr_nteeth").c_str(), s.serr_nteeth);
        pp.query((pre + "serr_duty").c_str(), s.serr_duty);

        v3.clear();
        pp.queryarr((pre + "ring_r").c_str(), v3);
        if (v3.size() == 2) { s.ring_r0 = v3[0]; s.ring_r1 = v3[1]; }

        // gear_rim parameters
        pp.query((pre + "rim_r_inner").c_str(), s.rim_r_inner);
        pp.query((pre + "rim_r_outer").c_str(), s.rim_r_outer);
        pp.query((pre + "rim_zc").c_str(), s.rim_zc);
        pp.query((pre + "rim_base_hy").c_str(), s.rim_base_hy);
        pp.query((pre + "rim_tooth_h").c_str(), s.rim_tooth_h);
        pp.query((pre + "rim_nteeth").c_str(), s.rim_nteeth);
        pp.query((pre + "rim_duty").c_str(), s.rim_duty);

        if ((s.shape == 0 || s.shape == 1) &&
            (s.hx <= Real(0.0) || s.hy <= Real(0.0) || s.hz <= Real(0.0)))
        {
            amrex::Abort(pre + "half must be three positive extents for bar/vane"
                         " (no cryo_solid_<n>_half found? indices must run 0..n_solids-1)");
        }
        if (s.shape == 2 && (s.serr_depth <= Real(0.0) || s.serr_nteeth <= 0))
        {
            amrex::Abort(pre + "serration needs serr_depth > 0 and serr_nteeth > 0");
        }
        if (s.shape == 3 &&
            (s.ring_r0 <= Real(0.0) || s.ring_r1 <= s.ring_r0 || s.hy <= Real(0.0)))
        {
            amrex::Abort(pre + "ring needs ring_r = r0 r1 with r1 > r0 > 0 and half = hx hy hz with hy > 0");
        }
        if (s.shape == 4 &&
            (s.rim_r_inner < Real(0.0) || s.rim_r_outer <= s.rim_r_inner ||
             s.rim_base_hy <= Real(0.0) || s.rim_tooth_h <= Real(0.0) || s.rim_nteeth <= 0))
        {
            amrex::Abort(pre + "gear_rim needs rim_r_inner >= 0, rim_r_outer > rim_r_inner, "
                               "rim_base_hy > 0, rim_tooth_h > 0, rim_nteeth > 0");
        }
    }
}

void incflo::cryo_apply_solids(Real x, Real y, Real z,
                               Real &velx, Real &vely, Real &velz,
                               int &cell_type_ijk,
                               Real time,
                               Real velz_plunge, Real plunge_disp) const
{
    Real const zp = z - plunge_disp; // co-moving frame

    for (int n = 0; n < m_cryo_n_solids; ++n)
    {
        auto const &s = m_cryo_solids[n];

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
                Real const c = std::cos(s.angle);
                Real const sn = std::sin(s.angle);
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

void incflo::cryo_set_geom_velocity(int i, int j, int k,
                                    Real x, Real y, Real z,
                                    Real &velx, Real &vely, Real &velz,
                                    int &cell_type_ijk,
                                    Real time,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_geom_velocity");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_update: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)

    // ***************************************************************
    // Plunging protocols
    // ***************************************************************

    Real velz_plunge, plunge_disp;
    cryo_plunge_state(time, velz_plunge, plunge_disp);

    // ***************************************************************
    // Set geometry and velocity based on the plunging protocol
    // ***************************************************************

    if (cryo_grid::GridGeom const *grid = cryo_grid::read_grid_geom(m_cryo_geometry))
    {
        // Disk family (gold EM grid / sapphire / diamond): a thin disk in the
        // x-z plane, normal along y, plunged edge-on. The disk body and its
        // optional +y-face sample layer co-move, so share the plunge velocity.
        // See cryo_grid.H for the per-material table.
        Real const init_z = (m_cryo_disk_init_z >= Real(0.0)) ? m_cryo_disk_init_z : grid->radius;
        Real const zoff = init_z + plunge_disp;
        if (!cryo_grid::apply_disk(*grid, x, y, z, zoff,
                                   m_cryo_sample_layer, m_cryo_sample_layer_thickness,
                                   cell_type_ijk, velx, vely, velz, velz_plunge))
        {
            cell_type_ijk = -1;
        }
    }
    else if (m_cryo_geometry == -1)
    {
        // -1: debug sphere, static
        Real Rdebug = 0.2;
        Real zoff = -2. * Rdebug + plunge_disp;
        Real geom_sphere = x * x + y * y + (z - zoff) * (z - zoff);
        if (geom_sphere < Rdebug * Rdebug)
        {
            cell_type_ijk = -6;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = velz_plunge;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }
    else if (m_cryo_geometry == 10)
    {
        // 10: Thermocouple probe, 60 um bead diameter with 25 um wire.
        // The spline-based motion is time-only, so compute it once here and
        // then apply the same kinematics to every cell in the probe body.
        cryo_tc::ThermocoupleMotion const motion = cryo_tc::evaluate_motion(time);

        // The existing cryo setup uses cell_type -2 for thermocouple material.
        // Model the bead as a sphere centered on the plunge path.
        Real const rtc1 = Real(0.03); // 60 um diameter -> 0.03 mm radius
        Real const zz = z - rtc1 + motion.depth;
        Real const geometry = x * x + y * y + zz * zz;

        if (geometry < rtc1 * rtc1)
        {
            cell_type_ijk = -2;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = -motion.speed;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }
    else if (m_cryo_geometry == 11)
    {
        // 11: Thermocouple probe, 100 um bead diameter with 25 um wire.
        // The spline-based motion is time-only, so compute it once here and
        // then apply the same kinematics to every cell in the probe body.
        cryo_tc::ThermocoupleMotion const motion = cryo_tc::evaluate_motion(time);

        // The existing cryo setup uses cell_type -2 for thermocouple material.
        // Model the bead as a sphere centered on the plunge path.
        Real const rtc1 = Real(0.0425); // 85 um diameter -> 0.0425 mm radius
        Real const zz = z - rtc1 + motion.depth;
        Real const geometry = x * x + y * y + zz * zz;

        if (geometry < rtc1 * rtc1)
        {
            cell_type_ijk = -2;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = -motion.speed;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }
    else if (m_cryo_geometry == 12)
    {
        // 12: Thermocouple probe, 400 um bead diameter with 127 um wire.
        // The spline-based motion is time-only, so compute it once here and
        // then apply the same kinematics to every cell in the probe body.
        cryo_tc::ThermocoupleMotion const motion = cryo_tc::evaluate_motion(time);

        // The existing cryo setup uses cell_type -2 for thermocouple material.
        // Model the bead as a sphere centered on the plunge path.
        Real const rtc1 = Real(0.21); // 420 um diameter -> 0.21 mm radius
        Real const zz = z - rtc1 + motion.depth;
        Real const geometry = x * x + y * y + zz * zz;

        if (geometry < rtc1 * rtc1)
        {
            cell_type_ijk = -2;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = -motion.speed;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }

#endif
}

void incflo::cryo_set_thermal(int i, int j, int k,
                              Real x, Real y, Real z,
                              int cell_type_ijk,
                              Real time,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_thermal");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_set_thermal: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)
    //
    return;
#endif
}

void incflo::cryo_set_temp_top_bc(Real z, Real &temperature_ijk, int cell_type_ijk,
                                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_temp_top_bc");

#if (AMREX_SPACEDIM == 2)
    amrex::Abort("cryo_set_temp_top_bc: not implemented in 2D");

#elif (AMREX_SPACEDIM == 3)
    // TODO: add a check on the m_cryo_geometry type
    if (z >= probhi[2] - dx[2] && cell_type_ijk != -1)
    {
        // Set top temperature boundary condition for solid cells
        temperature_ijk = m_cryo_temp_entry;
    }
    //  else {
    //     // Set top temperature boundary condition for fluid cells
    //     temperature_ijk = m_cryo_temp_eth;
    // }
#endif
}

#endif