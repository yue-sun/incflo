#include <incflo.H>

#include <fstream>
#include <iomanip>
#include <vector>

using namespace amrex;

void incflo::update_energy(StepType step_type, Vector<MultiFab> &scratch)
{
    BL_PROFILE("incflo::update_energy");

    if (!m_use_temperature)
    {
        return;
    }

    constexpr Real m_half = Real(0.5);
    Real l_dt = m_dt;

    // Thermal mass vhc = rho_mat(cell_type)*cp(T); rho from the same cell_type snapshot
    // compute_cp uses, decoupled from the uniform hydro density.
    Vector<MultiFab> rho_th(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        rho_th[lev].define(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
        compute_rho_th(lev, rho_th[lev]);
    }

    for (int lev = 0; lev <= finest_level; lev++)
    {
        auto &ld = *m_leveldata[lev];
        compute_cp(lev, ld.cp);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.tracer, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            Array4<Real const> const &tem_o = ld.temperature_o.const_array(mfi);
            Array4<Real> const &tem = ld.temperature.array(mfi);
            Array4<Real const> const rho_h = rho_th[lev].const_array(mfi);
            Array4<Real const> const &dtdt_o = ld.conv_temperature_o.const_array(mfi);
            // temperature forcing term (Q) is in scratch
            Array4<Real> const &tem_f = scratch[lev].array(mfi);
            Array4<Real const> const &cp = ld.cp.const_array(mfi);

            if (step_type == StepType::Corrector)
            {
                Array4<Real const> const &dtdt_n = ld.conv_temperature.const_array(mfi);

                if (m_diff_type == DiffusionType::Explicit)
                {
                    Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);
                    Array4<Real const> const &laps_n = ld.laps_tem.const_array(mfi);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                { tem(i, j, k) = tem_o(i, j, k) + l_dt *
                                                                      (m_half * (dtdt_o(i, j, k) + dtdt_n(i, j, k)) + (tem_f(i, j, k) + m_half * (laps_o(i, j, k) + laps_n(i, j, k))) / (rho_h(i, j, k) * cp(i, j, k))); });
                }
                else if (m_diff_type == DiffusionType::Crank_Nicolson)
                {
                    Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                {
                        tem(i,j,k) = tem_o(i,j,k) + l_dt *
                            ( m_half * (dtdt_o(i,j,k) + dtdt_n(i,j,k))
                              + (tem_f(i,j,k) + m_half * laps_o(i,j,k))
                                / (rho_h(i,j,k) * cp(i,j,k)) );
                        tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k); });
                }
                else if (m_diff_type == DiffusionType::Implicit)
                {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                {
                        tem(i,j,k) = tem_o(i,j,k) + l_dt *
                            ( m_half * (dtdt_o(i,j,k) + dtdt_n(i,j,k))
                              + tem_f(i,j,k) / (rho_h(i,j,k) * cp(i,j,k)) );
                        tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k); });
                }
            }
            else
            {
                // fall through to the predictor-style update below

                if (m_diff_type == DiffusionType::Explicit)
                {
                    Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                { tem(i, j, k) = tem_o(i, j, k) + l_dt *
                                                                      (dtdt_o(i, j, k) + (tem_f(i, j, k) + laps_o(i, j, k)) / (rho_h(i, j, k) * cp(i, j, k))); });
                }
                else if (m_diff_type == DiffusionType::Crank_Nicolson)
                {
                    Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                {
                    tem(i,j,k) = tem_o(i,j,k) + l_dt *
                        ( dtdt_o(i,j,k) + (tem_f(i,j,k) + m_half*laps_o(i,j,k))/(rho_h(i,j,k) * cp(i,j,k)) );
                    // Save rhoCp for use in implicit solve.
                    // Reuse scratch space since we are done with forcing now.
                    tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k); });
                }
                else if (m_diff_type == DiffusionType::Implicit)
                {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                {
                    // Note: only the forcing is divided by vhc; the advective
                    // tendency dtdt is already a temperature tendency (u·grad T).
                    tem(i,j,k) = tem_o(i,j,k) + l_dt *
                        (dtdt_o(i,j,k) + tem_f(i,j,k) / (rho_h(i,j,k) * cp(i,j,k)));
                    // Save rhoCp for use in implicit solve.
                    // Reuse scratch space since we are done with forcing now.
                    tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k); });
                }
            }
        } // mfi
    } // lev
}

// Total thermal energy sum(vhc*T*dV) over the AMR hierarchy (cells covered by
// a finer level are masked out). Diagnostic only: the re-stamped material map
// and the top temperature BC inject/remove energy, so this monitors trends and
// catches drift from the diffusion step, not a closed budget.
Real incflo::compute_thermal_energy()
{
    BL_PROFILE("incflo::compute_thermal_energy");

    Real energy = Real(0.0);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto &ld = *m_leveldata[lev];
        compute_cp(lev, ld.cp);
        MultiFab vhc(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
        compute_rho_th(lev, vhc);
        MultiFab::Multiply(vhc, ld.cp, 0, 0, 1, 0); // vhc = rho_mat * cp(T)

        iMultiFab level_mask;
        if (lev < finest_level)
        {
            level_mask = makeFineMask(grids[lev], dmap[lev], grids[lev + 1],
                                      refRatio(lev), 1, 0);
        }
        else
        {
            level_mask.define(grids[lev], dmap[lev], 1, 0);
            level_mask.setVal(1);
        }

        auto const &dx = geom[lev].CellSizeArray();
        Real const cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

        energy += cell_vol *
                  amrex::ReduceSum(vhc, ld.temperature, level_mask, 0,
                                   [=] AMREX_GPU_HOST_DEVICE(Box const &bx,
                                                             Array4<Real const> const &vhc_arr,
                                                             Array4<Real const> const &t_arr,
                                                             Array4<int const> const &mask_arr) -> Real
                                   {
                                       Real e = Real(0.0);
                                       amrex::Loop(bx, [=, &e](int i, int j, int k) noexcept
                                                   { e += mask_arr(i, j, k) * vhc_arr(i, j, k) * t_arr(i, j, k); });
                                       return e;
                                   });
    }

    ParallelDescriptor::ReduceRealSum(energy);
    return energy;
}

// Per-cell_type temperature stats (volume-weighted avg, min, max) over the AMR
// hierarchy (fine-covered cells masked out), one row per cell_type per step
// appended to a text file Python can read. The tracked types are fixed:
// material codes -1..-7 plus sample labels 1..m_cryo_n_samples. Types absent
// from the domain (e.g. a sample disk that hasn't plunged in yet) are reported
// at the entry temperature m_cryo_temp_entry.
void incflo::write_temperature_stats()
{
    BL_PROFILE("incflo::write_temperature_stats");

    if (!m_use_temperature)
    {
        return;
    }
#ifdef INCFLO_SIM_CRYO
    if (!m_sim_cryo)
    {
        return;
    }
#else
    return;
#endif

    constexpr int N_MAT = 7;                  // material codes -1..-7
    const int nsamp = m_cryo_n_samples;       // sample labels 1..nsamp
    const int ntypes = N_MAT + nsamp;
    constexpr Real BIG = Real(1.0e30);
    // dense index: code c<0 -> -c-1 (0..6); sample label c>=1 -> N_MAT+(c-1)

    Gpu::DeviceVector<Real> d_sumTvol(ntypes), d_sumvol(ntypes),
        d_ncell(ntypes), d_tmin(ntypes), d_tmax(ntypes);
    Real *p_sumTvol = d_sumTvol.dataPtr();
    Real *p_sumvol = d_sumvol.dataPtr();
    Real *p_ncell = d_ncell.dataPtr();
    Real *p_tmin = d_tmin.dataPtr();
    Real *p_tmax = d_tmax.dataPtr();
    amrex::ParallelFor(ntypes, [=] AMREX_GPU_DEVICE(int n) noexcept
                       {
        p_sumTvol[n] = Real(0.0); p_sumvol[n] = Real(0.0);
        p_ncell[n]   = Real(0.0); p_tmin[n]   = BIG; p_tmax[n] = -BIG; });

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto &ld = *m_leveldata[lev];

        iMultiFab level_mask;
        if (lev < finest_level)
        {
            level_mask = makeFineMask(grids[lev], dmap[lev], grids[lev + 1],
                                      refRatio(lev), 1, 0);
        }
        else
        {
            level_mask.define(grids[lev], dmap[lev], 1, 0);
            level_mask.setVal(1);
        }

        auto const &dx = geom[lev].CellSizeArray();
        Real const cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.temperature, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.tilebox();
            Array4<Real const> const &T = ld.temperature.const_array(mfi);
            Array4<int const> const &ct = ld.cell_type.const_array(mfi);
            Array4<int const> const &msk = level_mask.const_array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                if (msk(i,j,k) == 0) return;          // covered by finer level
                int c = ct(i,j,k), idx;
                if (c < 0) {
                    if (c < -N_MAT) return;            // unknown negative code
                    idx = -c - 1;
                } else if (c >= 1 && c <= nsamp) {
                    idx = N_MAT + (c - 1);
                } else {
                    return;                            // c==0 / out of range
                }
                Real t = T(i,j,k);
                Gpu::Atomic::Add(&p_sumTvol[idx], t * cell_vol);
                Gpu::Atomic::Add(&p_sumvol [idx], cell_vol);
                Gpu::Atomic::Add(&p_ncell  [idx], Real(1.0));
                Gpu::Atomic::Min(&p_tmin   [idx], t);
                Gpu::Atomic::Max(&p_tmax   [idx], t); });
        }
    }

    std::vector<Real> sumTvol(ntypes), sumvol(ntypes), ncell(ntypes),
        tmin(ntypes), tmax(ntypes);
    Gpu::copy(Gpu::deviceToHost, d_sumTvol.begin(), d_sumTvol.end(), sumTvol.begin());
    Gpu::copy(Gpu::deviceToHost, d_sumvol.begin(), d_sumvol.end(), sumvol.begin());
    Gpu::copy(Gpu::deviceToHost, d_ncell.begin(), d_ncell.end(), ncell.begin());
    Gpu::copy(Gpu::deviceToHost, d_tmin.begin(), d_tmin.end(), tmin.begin());
    Gpu::copy(Gpu::deviceToHost, d_tmax.begin(), d_tmax.end(), tmax.begin());

    ParallelDescriptor::ReduceRealSum(sumTvol.data(), ntypes);
    ParallelDescriptor::ReduceRealSum(sumvol.data(), ntypes);
    ParallelDescriptor::ReduceRealSum(ncell.data(), ntypes);
    ParallelDescriptor::ReduceRealMin(tmin.data(), ntypes);
    ParallelDescriptor::ReduceRealMax(tmax.data(), ntypes);

    if (!ParallelDescriptor::IOProcessor())
    {
        return;
    }

    // Append across runs/restarts; write the header only when starting a
    // fresh (nonexistent or empty) file.
    bool need_header = true;
    {
        std::ifstream test(m_temp_stats_file);
        if (test.good() && test.peek() != std::ifstream::traits_type::eof())
        {
            need_header = false;
        }
    }
    std::ofstream ofs(m_temp_stats_file, std::ios::out | std::ios::app);
    if (need_header)
    {
        ofs << "# step time cell_type n_cells avg_T min_T max_T\n";
    }
    ofs << std::setprecision(10);

    for (int idx = 0; idx < ntypes; ++idx)
    {
        int code = (idx < N_MAT) ? -(idx + 1) : (idx - N_MAT + 1);
        Real avg, mn, mx;
        if (ncell[idx] > Real(0.5))
        {
            avg = sumTvol[idx] / sumvol[idx];
            mn = tmin[idx];
            mx = tmax[idx];
        }
        else
        {
            avg = mn = mx = m_cryo_temp_entry; // not yet in cryogen
        }
        ofs << m_nstep << ' ' << m_cur_time << ' ' << code << ' '
            << static_cast<long long>(ncell[idx] + Real(0.5)) << ' '
            << avg << ' ' << mn << ' ' << mx << '\n';
    }
    ofs.close();
}

void incflo::update_temperature(StepType step_type, Vector<MultiFab *> const &tem_eta, Vector<MultiFab> &scratch)
{
    BL_PROFILE("incflo::update_temperature");

    if (!m_use_temperature)
    {
        return;
    }

    Vector<MultiFab const *> tem_eta_const;
    tem_eta_const.reserve(tem_eta.size());
    for (auto *mf : tem_eta)
    {
        tem_eta_const.push_back(mf);
    }

    Real const new_time = m_cur_time + m_dt;
    Real const half_time = m_cur_time + m_dt / 2.;

    // *************************************************************************************
    // Compute the temperature forcing terms
    // *************************************************************************************
    compute_tem_forces(half_time, GetVecOfPtrs(scratch));

    // *************************************************************************************
    // Compute explicit diffusive term (if corrector)
    // *************************************************************************************
    if (step_type == StepType::Corrector)
    {
        compute_temperature_diff_coeff(new_time, tem_eta);
        copy_from_new_to_old_thermal_conductivity();
        if (m_diff_type == DiffusionType::Explicit)
        {
            compute_laps_T(get_laps_tem_new(), get_temperature_new_const(), tem_eta_const);
        }
    }

    // *************************************************************************************
    // Update the temperature with time-explicit terms
    // *************************************************************************************
    update_energy(step_type, scratch);

    // *************************************************************************************
    // Solve implicit diffusion equation for temperature
    // *************************************************************************************
    if (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
    {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            fillphysbc_temperature(lev, new_time, m_leveldata[lev]->temperature, ng_diffusion);
        }
        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_dt : Real(0.5) * m_dt;

        // scratch holds vhc = rho_mat*cp(T); implicit solve uses chi = vhc, eta = k(T).
        diffuse_temperature(get_temperature_new(), GetVecOfPtrs(scratch), tem_eta_const,
                            dt_diff);
    }
    else
    {
        // Need to average down temperature since the diffusion solver didn't do it for us.
        for (int lev = finest_level - 1; lev >= 0; --lev)
        {
#ifdef AMREX_USE_EB
            amrex::EB_average_down(m_leveldata[lev + 1]->temperature, m_leveldata[lev]->temperature,
                                   0, 1, refRatio(lev));
#else
            amrex::average_down(m_leveldata[lev + 1]->temperature, m_leveldata[lev]->temperature,
                                0, 1, refRatio(lev));
#endif
        }
    }
}
