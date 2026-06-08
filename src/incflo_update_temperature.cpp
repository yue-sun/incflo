#include <incflo.H>

using namespace amrex;

void incflo::update_energy(StepType step_type, Vector<MultiFab> &scratch)
{
    BL_PROFILE("incflo::update_energy");

    if (!m_use_temperature)
    {
        return;
    }

    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

    constexpr Real m_half = Real(0.5);
    Real l_dt = m_dt;

    Vector<MultiFab> energy(finest_level + 1);
    if (conservative_temperature)
    {
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            energy[lev].define(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
        }
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
            Array4<Real const> const &rho_o = ld.density_o.const_array(mfi);
            Array4<Real const> const &rho_h = ld.density_nph.const_array(mfi);
            Array4<Real const> const &dtdt_o = ld.conv_temperature_o.const_array(mfi);
            // temperature forcing term (Q) is in scratch
            Array4<Real> const &tem_f = scratch[lev].array(mfi);
            Array4<Real const> const &cp = ld.cp.const_array(mfi);

            if (conservative_temperature)
            {
                Array4<Real> const &ener = energy[lev].array(mfi);

                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                            { ener(i, j, k) = rho_o(i, j, k) * cp(i, j, k) * tem_o(i, j, k); });
            }

            if (step_type == StepType::Corrector)
            {
                Array4<Real const> const &dtdt_n = ld.conv_temperature.const_array(mfi);

                if (conservative_temperature)
                {
                    Array4<Real> const &ener = energy[lev].array(mfi);
                    // TODO: check if this is needed
                    constexpr Real rhocp_floor = Real(1.0e-12);

                    if (m_diff_type == DiffusionType::Explicit)
                    {
                        Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);
                        Array4<Real const> const &laps_n = ld.laps_tem.const_array(mfi);
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                            Real ener_new = ener(i,j,k) + l_dt *
                                ( m_half * rho_h(i,j,k) * cp(i,j,k) * (dtdt_o(i,j,k) + dtdt_n(i,j,k))
                                  + tem_f(i,j,k)
                                  + m_half * (laps_o(i,j,k) + laps_n(i,j,k)) );
                            Real rhocp = amrex::max(amrex::Math::abs(rho_h(i,j,k) * cp(i,j,k)), rhocp_floor);
                            tem(i,j,k) = ener_new / rhocp;
                            ener(i,j,k) = ener_new; });
                    }
                    else if (m_diff_type == DiffusionType::Crank_Nicolson)
                    {
                        Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                            Real ener_new = ener(i,j,k) + l_dt *
                                ( m_half * rho_h(i,j,k) * cp(i,j,k) * (dtdt_o(i,j,k) + dtdt_n(i,j,k))
                                  + tem_f(i,j,k)
                                  + m_half * laps_o(i,j,k) );
                            Real rhocp = amrex::max(amrex::Math::abs(rho_h(i,j,k) * cp(i,j,k)), rhocp_floor);
                            tem(i,j,k) = ener_new / rhocp;
                            // Save rhoCp for use in implicit solve.
                            tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k);
                            ener(i,j,k) = ener_new; });
                    }
                    else if (m_diff_type == DiffusionType::Implicit)
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                            Real ener_new = ener(i,j,k) + l_dt *
                                ( m_half * rho_h(i,j,k) * cp(i,j,k) * (dtdt_o(i,j,k) + dtdt_n(i,j,k))
                                  + tem_f(i,j,k) );
                            Real rhocp = amrex::max(amrex::Math::abs(rho_h(i,j,k) * cp(i,j,k)), rhocp_floor);
                            tem(i,j,k) = ener_new / rhocp;
                            // Save rhoCp for use in implicit solve.
                            tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k);
                            ener(i,j,k) = ener_new; });
                    }
                }
                else
                {
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
            }
            else
            {
                // fall through to the predictor-style update below

                if (m_diff_type == DiffusionType::Explicit)
                {
                    Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);

                    if (conservative_temperature)
                    {
                        Array4<Real> const &ener = energy[lev].array(mfi);
                        constexpr Real rhocp_floor = Real(1.0e-12);

                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                        Real ener_new = ener(i,j,k) + l_dt *
                            ( rho_h(i,j,k) * cp(i,j,k) * dtdt_o(i,j,k)
                              + tem_f(i,j,k) + laps_o(i,j,k) );
                        Real rhocp = amrex::max(amrex::Math::abs(rho_h(i,j,k) * cp(i,j,k)), rhocp_floor);
                        tem(i,j,k) = ener_new / rhocp;
                        ener(i,j,k) = ener_new; });
                    }
                    else
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    { tem(i, j, k) = tem_o(i, j, k) + l_dt *
                                                                          (dtdt_o(i, j, k) + (tem_f(i, j, k) + laps_o(i, j, k)) / (rho_h(i, j, k) * cp(i, j, k))); });
                    }
                }
                else if (m_diff_type == DiffusionType::Crank_Nicolson)
                {
                    Array4<Real const> const &laps_o = ld.laps_tem_o.const_array(mfi);

                    if (conservative_temperature)
                    {
                        Array4<Real> const &ener = energy[lev].array(mfi);
                        constexpr Real rhocp_floor = Real(1.0e-12);

                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                        Real ener_new = ener(i,j,k) + l_dt *
                            ( rho_h(i,j,k) * cp(i,j,k) * dtdt_o(i,j,k)
                              + tem_f(i,j,k) + m_half * laps_o(i,j,k) );
                        Real rhocp = amrex::max(amrex::Math::abs(rho_h(i,j,k) * cp(i,j,k)), rhocp_floor);
                        tem(i,j,k) = ener_new / rhocp;
                        // Save rhoCp for use in implicit solve.
                        // Reuse scratch space since we are done with forcing now.
                        tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k);
                        ener(i,j,k) = ener_new; });
                    }
                    else
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                        tem(i,j,k) = tem_o(i,j,k) + l_dt *
                            ( dtdt_o(i,j,k) + (tem_f(i,j,k) + m_half*laps_o(i,j,k))/(rho_h(i,j,k) * cp(i,j,k)) );
                        // Save rhoCp for use in implicit solve.
                        // Reuse scratch space since we are done with forcing now.
                        tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k); });
                    }
                }
                else if (m_diff_type == DiffusionType::Implicit)
                {
                    if (conservative_temperature)
                    {
                        Array4<Real> const &ener = energy[lev].array(mfi);
                        constexpr Real rhocp_floor = Real(1.0e-12);

                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                        Real ener_new = ener(i,j,k) + l_dt *
                            (rho_h(i,j,k) * cp(i,j,k) * dtdt_o(i,j,k) + tem_f(i,j,k));
                        Real rhocp = amrex::max(amrex::Math::abs(rho_h(i,j,k) * cp(i,j,k)), rhocp_floor);
                        tem(i,j,k) = ener_new / rhocp;
                        // Save rhoCp for use in implicit solve.
                        // Reuse scratch space since we are done with forcing now.
                        tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k);
                        ener(i,j,k) = ener_new; });
                    }
                    else
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                    {
                        tem(i,j,k) = tem_o(i,j,k) + l_dt *
                            (dtdt_o(i,j,k) + tem_f(i,j,k)) / (rho_h(i,j,k) * cp(i,j,k));
                        // Save rhoCp for use in implicit solve.
                        // Reuse scratch space since we are done with forcing now.
                        tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k); });
                    }
                }
            }
        } // mfi
    } // lev

    if (conservative_temperature)
    {
        for (int lev = finest_level - 1; lev >= 0; --lev)
        {
#ifdef AMREX_USE_EB
            amrex::EB_average_down(energy[lev + 1], energy[lev], 0, 1, refRatio(lev));
#else
            amrex::average_down(energy[lev + 1], energy[lev], 0, 1, refRatio(lev));
#endif
        }
    }
}

void incflo::update_temperature(StepType step_type, Vector<MultiFab *> const &tem_eta, Vector<MultiFab> &scratch)
{
    BL_PROFILE("incflo::update_temperature");

    if (!m_use_temperature)
    {
        return;
    }

    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

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

        // scratch holds rhoCp; solve directly for T with chi=rhoCp for both modes.
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
