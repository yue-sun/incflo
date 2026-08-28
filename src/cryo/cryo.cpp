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

    // Time-only kinematics and scalar inputs, evaluated ONCE here instead of
    // per cell: cryo_plunge_state walks the prescribed-protocol Vectors and
    // cryo_tc::evaluate_motion is two 21-segment spline searches, and both
    // depend on `time` alone. Everything the per-cell path needs is then POD
    // (cryo_stamp::DiskParams / SampleData + a device pointer for the solids),
    // so the kernel below never dereferences `this`.
    Real velz_plunge, plunge_disp;
    cryo_plunge_state(time, velz_plunge, plunge_disp);
    cryo_tc::ThermocoupleMotion const tc_motion = cryo_tc::evaluate_motion(time);

    cryo_stamp::DiskParams const disk = cryo_disk_params(velz_plunge, plunge_disp, tc_motion);
    cryo_stamp::SampleData const samples = cryo_sample_data(m_cryo_geometry, plunge_disp);
    cryo_stamp::Solid const* solids_p = m_cryo_solids_d.dataPtr();
    int const n_solids = m_cryo_n_solids;
    Real const temp_entry = m_cryo_temp_entry;

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
            Array4<int> const &cell_type = ld.cell_type.array(mfi);
            Array4<Real> const &temperature = ld.temperature.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
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
                            cryo_stamp::set_geom_velocity(x, y, z, velx, vely, velz,
                                                          cell_type_ijk, disk);
                            // Additive wiper/obstacle primitives
                            if (n_solids > 0)
                            {
                                cryo_stamp::apply_solids(x, y, z, velx, vely, velz,
                                                         cell_type_ijk, solids_p, n_solids,
                                                         time, disk.velz_plunge,
                                                         disk.plunge_disp);
                            }
                            // Discrete biological samples on the disk face
                            if (samples.host_valid)
                            {
                                cryo_stamp::apply_samples(x, y, z, velx, vely, velz,
                                                          cell_type_ijk, disk.velz_plunge,
                                                          samples);
                            }
                            // Set thermal properties (no-op hook) and the top
                            // temperature B.C.
                            cryo_stamp::set_thermal(x, y, z, cell_type_ijk, time);
                            cryo_stamp::set_temp_top_bc(z, temperature_ijk, cell_type_ijk,
                                                        dx, probhi, temp_entry);
                        });
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

// Resolve the geometry preset once per update: the disk-table row (if this
// geometry is a disk), its centre z for this time, and the time-only kinematics.
// The result is POD and is what the device stamper reads instead of members.
cryo_stamp::DiskParams incflo::cryo_disk_params (Real velz_plunge, Real plunge_disp,
                                                cryo_tc::ThermocoupleMotion const &motion) const
{
    cryo_stamp::DiskParams p;
    p.geometry = m_cryo_geometry;
    p.sample_layer = m_cryo_sample_layer ? 1 : 0;
    p.sample_layer_thickness = m_cryo_sample_layer_thickness;
    p.velz_plunge = velz_plunge;
    p.plunge_disp = plunge_disp;
    p.motion = motion;

    p.is_disk = cryo_grid::read_grid_geom(m_cryo_geometry, p.grid);
    if (p.is_disk)
    {
        Real const init_z = (m_cryo_disk_init_z >= Real(0.0)) ? m_cryo_disk_init_z
                                                             : p.grid.radius;
        p.zoff = init_z + plunge_disp;
    }
    return p;
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
        // Cache the rotation: `angle` is a parse-time constant, so computing
        // cos/sin per cell (as the stamper used to) is pure waste.
        s.cos_angle = std::cos(s.angle);
        s.sin_angle = std::sin(s.angle);

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

    // Mirror the parsed primitives into device memory once: the geometry is
    // fixed for the run, so the per-cell stamper only ever reads this array.
    // (On a CPU build this is an arena allocation in host memory.)
    m_cryo_solids_d.resize(m_cryo_solids.size());
    if (!m_cryo_solids.empty())
    {
        Gpu::copyAsync(Gpu::hostToDevice, m_cryo_solids.begin(), m_cryo_solids.end(),
                       m_cryo_solids_d.begin());
        Gpu::streamSynchronize();
    }
}

#endif
