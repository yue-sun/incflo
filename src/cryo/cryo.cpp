#include <incflo.H>
#include <cryo_tc_experiment.H>

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

        for (MFIter mfi(ld.density, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            Array4<Real> const &vel = ld.velocity.array(mfi);
            Array4<Real> const &density = ld.density.array(mfi);
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
                            Real &rho_ijk = density(i, j, k);


                            // Set geometry and velocity
                            cryo_set_geom_velocity(i, j, k, x, y, z,
                                                   velx, vely, velz,
                                                   cell_type_ijk,
                                                   rho_ijk,
                                               time, dx, problo, probhi);
                            // Additive wiper/obstacle primitives
                            if (m_cryo_n_solids > 0)
                            {
                                cryo_apply_solids(x, y, z,
                                                  velx, vely, velz,
                                                  cell_type_ijk, rho_ijk,
                                                  time, velz_plunge, plunge_disp);
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
        else { amrex::Abort(pre + "shape must be bar, vane, or serration"); }

        pp.query((pre + "material").c_str(), s.material);
        if (s.material != -2 && s.material != -3 && s.material != -4 &&
            s.material != -5 && s.material != -6)
        {
            amrex::Abort(pre + "material must be a solid cell_type (-2..-6)");
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

        if (s.shape != 2 && (s.hx <= Real(0.0) || s.hy <= Real(0.0) || s.hz <= Real(0.0)))
        {
            amrex::Abort(pre + "half must be three positive extents for bar/vane");
        }
        if (s.shape == 2 && (s.serr_depth <= Real(0.0) || s.serr_nteeth <= 0))
        {
            amrex::Abort(pre + "serration needs serr_depth > 0 and serr_nteeth > 0");
        }
    }
}

void incflo::cryo_apply_solids(Real x, Real y, Real z,
                               Real &velx, Real &vely, Real &velz,
                               int &cell_type_ijk, Real &rho_ijk,
                               Real time,
                               Real velz_plunge, Real plunge_disp) const
{
    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

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
                if (conservative_temperature) { rho_ijk = cryo_props::rho_eth; }
            }
            continue;
        }

        // bar / vane: box test in the (possibly oscillation-shifted, possibly
        // rotated) co-moving frame
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
        Real pz = zp - (s.cz + offz);
        if (s.angle != Real(0.0))
        {
            Real const c = std::cos(s.angle);
            Real const sn = std::sin(s.angle);
            Real const qx = c * px + sn * pz;
            Real const qz = -sn * px + c * pz;
            px = qx;
            pz = qz;
        }

        if (amrex::Math::abs(px) < s.hx &&
            amrex::Math::abs(py) < s.hy &&
            amrex::Math::abs(pz) < s.hz)
        {
            cell_type_ijk = s.material;
            velx = voscx;
            vely = Real(0.0);
            velz = velz_plunge + voscz;
            if (conservative_temperature)
            {
                if (s.material == -2)      { rho_ijk = cryo_props::rho_tcp; }
                else if (s.material == -4) { rho_ijk = cryo_props::rho_sap; }
                else if (s.material == -5) { rho_ijk = cryo_props::rho_dia; }
                else                       { rho_ijk = cryo_props::rho_plu; } // -3, -6
            }
        }
    }
}

void incflo::cryo_set_geom_velocity(int i, int j, int k,
                                    Real x, Real y, Real z,
                                    Real &velx, Real &vely, Real &velz,
                                    int &cell_type_ijk,
                                    Real &rho_ijk,
                                    Real time,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi)
{
    BL_PROFILE("incflo::cryo_set_geom_velocity");

    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

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

    if (m_cryo_geometry == -1)
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
    else if (m_cryo_geometry == 4)
    {
        // -4: sapphire disk
        Real R_sap_disk = Real(1.5);      // sapphire disk radius: 1.5mm
        Real w_sap_disk = Real(0.16 / 2); // sapphire disk width: 160um

        Real zoff = R_sap_disk + plunge_disp;
        Real geom_sap_disk = x * x + (z - zoff) * (z - zoff);
        Real geom_sap_disk_thickness = amrex::Math::abs(y);
        // Disk geometry, plus an optional sample layer (water properties)
        // coating its +y face; both move with the disk, so share kinematics.
        bool const in_disk = geom_sap_disk_thickness < w_sap_disk;
        bool const in_sample =
            m_cryo_sample_layer && y >= w_sap_disk &&
            y < w_sap_disk + m_cryo_sample_layer_thickness;
        if (geom_sap_disk < R_sap_disk * R_sap_disk && (in_disk || in_sample))
        {
            cell_type_ijk = in_disk ? -4 : 0;
            velx = Real(0.0);
            vely = Real(0.0);
            velz = velz_plunge;
        }
        else
        {
            cell_type_ijk = -1;
        }
    }
    else if (m_cryo_geometry == 5)
    {
        // -5: diamond disk
        Real R_dia_disk = Real(1.5);     // diamond disk radius: 1.5mm
        Real w_dia_disk = Real(0.1 / 2); // diamond disk width: 0.1mm

        Real zoff = R_dia_disk + plunge_disp;
        Real geom_dia_disk = x * x + (z - zoff) * (z - zoff);
        Real geom_dia_disk_thickness = amrex::Math::abs(y);
        // Disk geometry, plus an optional sample layer (water properties)
        // coating its +y face; both move with the disk, so share kinematics.
        bool const in_disk = geom_dia_disk_thickness < w_dia_disk;
        bool const in_sample =
            m_cryo_sample_layer && y >= w_dia_disk &&
            y < w_dia_disk + m_cryo_sample_layer_thickness;
        if (geom_dia_disk < R_dia_disk * R_dia_disk && (in_disk || in_sample))
        {
            cell_type_ijk = in_disk ? -5 : 0;
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

    if (conservative_temperature)
    {
        // Map material density from cell type only for conservative temperature mode.
        if (cell_type_ijk == -1)
        {
            rho_ijk = cryo_props::rho_eth;
        }
        else if (cell_type_ijk == -2)
        {
            rho_ijk = cryo_props::rho_tcp;
        }
        else if (cell_type_ijk == -3 || cell_type_ijk == -6)
        {
            rho_ijk = cryo_props::rho_plu;
        }
        else if (cell_type_ijk == -4)
        {
            rho_ijk = cryo_props::rho_sap;
        }
        else if (cell_type_ijk == -5)
        {
            rho_ijk = cryo_props::rho_dia;
        }
        else if (cell_type_ijk >= 0)
        {
            rho_ijk = cryo_props::rho_sam;
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