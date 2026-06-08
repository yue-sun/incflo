#include <incflo.H>

#include <cmath>
#include <limits>

using namespace amrex;

//
// Compute new dt by using the formula derived in
// "A Boundary Condition Capturing Method for Multiphase Incompressible Flow"
// by Kang et al. (JCP).
//
//  dt/2 * ( C+V + sqrt( (C+V)**2 + 4Fx/dx + 4Fy/dy + 4Fz/dz )
//
// where
//
// C = max(|U|)/dx + max(|V|)/dy + max(|W|)/dz    --> Convection
//
// V = 2 * max(eta/rho) * (1/dx^2 + 1/dy^2 +1/dz^2) --> Diffusion
//
// Fx, Fy, Fz = net acceleration due to external forces
//
// WARNING: We use a slightly modified version of C in the implementation below
//
void incflo::ComputeDt(int initialization, bool explicit_diffusion)
{
    BL_PROFILE("incflo::ComputeDt");

    // Store the past two dt
    m_prev_prev_dt = m_prev_dt;
    m_prev_dt = m_dt;

    Real conv_cfl = Real(0.0);
    Real diff_cfl = Real(0.0);
    Real forc_cfl = Real(0.0);
    Real therm_cfl = Real(0.0);
    bool apply_thermal_dt_constraint = true;
#ifdef INCFLO_SIM_CRYO
    int has_solid_cells = 0;
#endif

    bool const conservative_temperature =
        !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

    if (m_use_temperature && m_thermal_cfl > Real(0.0)) {
        compute_temperature_diff_coeff(m_cur_time, get_thermal_conductivity_new());
        if (conservative_temperature) {
            for (int lev = 0; lev <= finest_level; ++lev) {
                compute_cp(lev, m_leveldata[lev]->cp);
            }
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto const dxinv = geom[lev].InvCellSizeArray();
        MultiFab const &vel = m_leveldata[lev]->velocity;
        MultiFab const &rho = m_leveldata[lev]->density;
        MultiFab const &tra = m_leveldata[lev]->tracer;
        MultiFab const &tra_o = m_leveldata[lev]->tracer_o;

        Real conv_lev = Real(0.0);
        Real diff_lev = Real(0.0);
        Real forc_lev = Real(0.0);

        // Make a temporary here to hold vel_forces
        MultiFab vel_forces(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);

        compute_vel_forces_on_level(lev, vel_forces, vel, rho, tra_o, tra);

#ifdef AMREX_USE_EB
        if (!vel.isAllRegular())
        {
            auto const &flag = EBFactory(lev).getMultiEBCellFlagFab();
            conv_lev = amrex::ReduceMax(vel, flag, 0,
                                        [=] AMREX_GPU_HOST_DEVICE(Box const &b,
                                                                  Array4<Real const> const &v,
                                                                  Array4<EBCellFlag const> const &f) -> Real
                                        {
                                            Real mx = -1.0;
                                            amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                        {
                               if (!f(i,j,k).isCovered()) {
                                   mx = amrex::max(AMREX_D_DECL(amrex::Math::abs(v(i,j,k,0))*dxinv[0],
                                                                amrex::Math::abs(v(i,j,k,1))*dxinv[1],
                                                                amrex::Math::abs(v(i,j,k,2))*dxinv[2]), mx);
                               } });
                                            return mx;
                                        });
            if (explicit_diffusion)
            {
                diff_lev = amrex::ReduceMax(rho, flag, 0,
                                            [=] AMREX_GPU_HOST_DEVICE(Box const &b,
                                                                      Array4<Real const> const &r,
                                                                      Array4<EBCellFlag const> const &f) -> Real
                                            {
                                                Real mx = Real(-1.0);
                                                amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                            {
                                  if (!f(i,j,k).isCovered()) {
                                      Real rho_inv = Real(1.0)/r(i,j,k);
                                      mx = amrex::max(rho_inv, mx);
                                  } });
                                                return mx;
                                            });
                diff_lev *= m_mu;
            }

            // Forcing term -- old way of computing
            // const auto dxinv_finest = Geom(finest_level).InvCellSizeArray();
            // forc_lev = std::abs(m_gravity[0] - std::abs(m_gp0[0])) * dxinv_finest[0]
            //          + std::abs(m_gravity[1] - std::abs(m_gp0[1])) * dxinv_finest[1]
            //          + std::abs(m_gravity[2] - std::abs(m_gp0[2])) * dxinv_finest[2];

            // Forcing term -- new way of computing using "actual" forcing term
            forc_lev = amrex::ReduceMax(vel_forces, flag, 0,
                                        [=] AMREX_GPU_HOST_DEVICE(Box const &b,
                                                                  Array4<Real const> const &vf,
                                                                  Array4<EBCellFlag const> const &f) -> Real
                                        {
                                            Real mx = Real(-1.0);
                                            amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                        {
                          if (!f(i,j,k).isCovered()) {
                              mx = amrex::max(AMREX_D_DECL(amrex::Math::abs(vf(i,j,k,0))*dxinv[0],
                                                           amrex::Math::abs(vf(i,j,k,1))*dxinv[1],
                                                           amrex::Math::abs(vf(i,j,k,2))*dxinv[2]), mx);
                          } });
                                            return mx;
                                        });
        }
        else
#endif
        {
            conv_lev = amrex::ReduceMax(vel, 0,
                                        [=] AMREX_GPU_HOST_DEVICE(Box const &b,
                                                                  Array4<Real const> const &v) -> Real
                                        {
                                            Real mx = Real(-1.0);
                                            amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                        { mx = amrex::max(AMREX_D_DECL(amrex::Math::abs(v(i, j, k, 0)) * dxinv[0],
                                                                                       amrex::Math::abs(v(i, j, k, 1)) * dxinv[1],
                                                                                       amrex::Math::abs(v(i, j, k, 2)) * dxinv[2]),
                                                                          mx); });
                                            return mx;
                                        });

            if (explicit_diffusion)
            {
                diff_lev = amrex::ReduceMax(rho, 0,
                                            [=] AMREX_GPU_HOST_DEVICE(Box const &b,
                                                                      Array4<Real const> const &r) -> Real
                                            {
                                                Real mx = Real(-1.0);
                                                amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                            {
                                   Real rho_inv = Real(1.0)/r(i,j,k);
                                   mx = amrex::max(rho_inv, mx); });
                                                return mx;
                                            });
                diff_lev *= m_mu;
            }

            // Forcing term -- old way of computing
            // const auto dxinv_finest = Geom(finest_level).InvCellSizeArray();
            // forc_lev = std::abs(m_gravity[0] - std::abs(m_gp0[0])) * dxinv_finest[0]
            //          + std::abs(m_gravity[1] - std::abs(m_gp0[1])) * dxinv_finest[1]
            //          + std::abs(m_gravity[2] - std::abs(m_gp0[2])) * dxinv_finest[2];

            // Forcing term -- new way of computing using "actual" forcing term
            forc_lev = amrex::ReduceMax(vel_forces, 0,
                                        [=] AMREX_GPU_HOST_DEVICE(Box const &b,
                                                                  Array4<Real const> const &vf) -> Real
                                        {
                                            Real mx = Real(-1.0);
                                            amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                        { mx = amrex::max(AMREX_D_DECL(amrex::Math::abs(vf(i, j, k, 0)) * dxinv[0],
                                                                                       amrex::Math::abs(vf(i, j, k, 1)) * dxinv[1],
                                                                                       amrex::Math::abs(vf(i, j, k, 2)) * dxinv[2]),
                                                                          mx); });
                                            return mx;
                                        });
        }

        forc_cfl = std::max(forc_cfl, forc_lev);
        conv_cfl = std::max(conv_cfl, conv_lev);

#if (AMREX_SPACEDIM == 2)
        Real dxinv_norm = dxinv[0] * dxinv[0] + dxinv[1] * dxinv[1];
#else
        Real dxinv_norm = dxinv[0] * dxinv[0] + dxinv[1] * dxinv[1] + dxinv[2] * dxinv[2];
#endif

        diff_cfl = std::max(diff_cfl, diff_lev * Real(2.0) * dxinv_norm);

        if (m_use_temperature && m_thermal_cfl > Real(0.0))
        {
            Real therm_lev = Real(0.0);
            MultiFab const& eta_T = m_leveldata[lev]->thermal_conductivity;

#ifdef INCFLO_SIM_CRYO
            if (m_sim_cryo)
            {
                iMultiFab const& cell_type = m_leveldata[lev]->cell_type;

                int lev_has_nonfluid = amrex::ReduceMax(cell_type, 0,
                                                        [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                                  Array4<int const> const& ct) -> int
                                                        {
                                                            int mx = 0;
                                                            amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                            {
                                                                if (ct(i,j,k) != -1) {
                                                                    mx = 1;
                                                                }
                                                            });
                                                            return mx;
                                                        });

                if (lev_has_nonfluid != 0) {
                    has_solid_cells = 1;
                }

                Real alpha_ref = cryo_props::kappa_eth;
                if (lev_has_nonfluid != 0)
                {
                    int has_tcp = amrex::ReduceMax(cell_type, 0,
                                                   [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                             Array4<int const> const& ct) -> int
                                                   {
                                                       int mx = 0;
                                                       amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                       {
                                                           if (ct(i,j,k) == -2) { mx = 1; }
                                                       });
                                                       return mx;
                                                   });
                    int has_plu = amrex::ReduceMax(cell_type, 0,
                                                   [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                             Array4<int const> const& ct) -> int
                                                   {
                                                       int mx = 0;
                                                       amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                       {
                                                           if (ct(i,j,k) == -3 || ct(i,j,k) == -6) { mx = 1; }
                                                       });
                                                       return mx;
                                                   });
                    int has_sap = amrex::ReduceMax(cell_type, 0,
                                                   [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                             Array4<int const> const& ct) -> int
                                                   {
                                                       int mx = 0;
                                                       amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                       {
                                                           if (ct(i,j,k) == -4) { mx = 1; }
                                                       });
                                                       return mx;
                                                   });
                    int has_dia = amrex::ReduceMax(cell_type, 0,
                                                   [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                             Array4<int const> const& ct) -> int
                                                   {
                                                       int mx = 0;
                                                       amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                       {
                                                           if (ct(i,j,k) == -5) { mx = 1; }
                                                       });
                                                       return mx;
                                                   });
                    int has_sam = amrex::ReduceMax(cell_type, 0,
                                                   [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                             Array4<int const> const& ct) -> int
                                                   {
                                                       int mx = 0;
                                                       amrex::Loop(b, [=, &mx](int i, int j, int k) noexcept
                                                       {
                                                           if (ct(i,j,k) >= 0) { mx = 1; }
                                                       });
                                                       return mx;
                                                   });

                    alpha_ref = Real(0.0);
                    if (has_tcp != 0) alpha_ref = amrex::max(alpha_ref, cryo_props::kappa_tcp);
                    if (has_plu != 0) alpha_ref = amrex::max(alpha_ref, cryo_props::kappa_plu);
                    if (has_sap != 0) alpha_ref = amrex::max(alpha_ref, cryo_props::kappa_sap);
                    if (has_dia != 0) alpha_ref = amrex::max(alpha_ref, cryo_props::kappa_dia);
                    if (has_sam != 0) alpha_ref = amrex::max(alpha_ref, cryo_props::kappa_sam);
                    if (alpha_ref <= Real(0.0)) {
                        alpha_ref = cryo_props::kappa_eth;
                    }
                }

                therm_lev = alpha_ref;
            }
            else
#endif
            {
                if (conservative_temperature)
                {
                    MultiFab const& cp = m_leveldata[lev]->cp;
                    constexpr Real denom_floor = Real(1.0e-12);
                    constexpr Real alpha_cap = Real(1.0e300);

                    therm_lev = amrex::ReduceMax(eta_T, rho, cp, 0,
                                                 [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                           Array4<Real const> const& k,
                                                                           Array4<Real const> const& r,
                                                                           Array4<Real const> const& cp_a) -> Real
                                                 {
                                                     Real mx = Real(0.0);
                                                     amrex::Loop(b, [=, &mx](int i, int j, int kidx) noexcept
                                                     {
                                                         Real denom = amrex::max(amrex::Math::abs(r(i,j,kidx) * cp_a(i,j,kidx)), denom_floor);
                                                         Real alpha = amrex::max(Real(0.0), k(i,j,kidx)) / denom;
                                                         alpha = amrex::min(alpha, alpha_cap);
                                                         mx = amrex::max(mx, alpha);
                                                     });
                                                     return mx;
                                                 });
                }
                else
                {
                    constexpr Real alpha_cap = Real(1.0e300);
                    therm_lev = amrex::ReduceMax(eta_T, 0,
                                                 [=] AMREX_GPU_HOST_DEVICE(Box const& b,
                                                                           Array4<Real const> const& k) -> Real
                                                 {
                                                     Real mx = Real(0.0);
                                                     amrex::Loop(b, [=, &mx](int i, int j, int kidx) noexcept
                                                     {
                                                         Real alpha = amrex::min(amrex::max(Real(0.0), k(i,j,kidx)), alpha_cap);
                                                         mx = amrex::max(mx, alpha);
                                                     });
                                                     return mx;
                                                 });
                }
            }

            therm_cfl = std::max(therm_cfl, therm_lev * Real(2.0) * dxinv_norm);

        }
    }

    Real cd_cfl;
    if (explicit_diffusion)
    {
        ParallelAllReduce::Max<Real>({conv_cfl, diff_cfl},
                                     ParallelContext::CommunicatorSub());
        cd_cfl = conv_cfl + diff_cfl;
    }
    else
    {
        ParallelAllReduce::Max<Real>(conv_cfl,
                                     ParallelContext::CommunicatorSub());
        cd_cfl = conv_cfl;
    }

    ParallelAllReduce::Max<Real>(forc_cfl,
                                 ParallelContext::CommunicatorSub());

    if (m_use_temperature && m_thermal_cfl > Real(0.0)) {
        ParallelAllReduce::Max<Real>(therm_cfl,
                                     ParallelContext::CommunicatorSub());

#ifdef INCFLO_SIM_CRYO
        if (m_sim_cryo) {
            apply_thermal_dt_constraint = true;
        }
#endif
    }

    // Combined CFL conditioner
    Real comb_cfl = cd_cfl + std::sqrt(cd_cfl * cd_cfl + Real(4.0) * forc_cfl);

    // Update dt
    Real dt_new;
    if (comb_cfl > 0.)
    {
        dt_new = Real(2.0) * m_cfl / comb_cfl;
    }
    else
    {
        // This is totally random but just a way to set a timestep
        // when the initial velocity is zero and the forcing term
        // is not a body force
        auto const dx = geom[finest_level].CellSizeArray();
        dt_new = std::min(dx[0], dx[1]);
#if (AMREX_SPACEDIM == 3)
        dt_new = std::min(dt_new, dx[2]);
#endif
    }

    if (m_use_temperature && m_thermal_cfl > Real(0.0) && therm_cfl > Real(0.0) && apply_thermal_dt_constraint)
    {
        Real dt_thermal = m_thermal_cfl / therm_cfl;
        if (std::isfinite(dt_thermal) && dt_thermal > Real(0.0)) {
            dt_new = amrex::min(dt_new, dt_thermal);
        }
    }

    // Optionally reduce CFL for initial step
    if (initialization)
    {
        dt_new *= m_init_shrink;
    }

    // Protect against very small comb_cfl
    // This may happen, for example, when the initial velocity field
    // is zero for an inviscid flow with no external forcing
    Real eps = std::numeric_limits<Real>::epsilon();
    constexpr Real tiny_dt_threshold = Real(1.0e-8);
    if (!initialization && comb_cfl <= eps)
    {
        dt_new = Real(0.5) * m_dt;
        if (m_dt_min > Real(0.0)) {
            dt_new = amrex::max(dt_new, m_dt_min);
        }
        amrex::Print() << "WARNING: comb_cfl is very small (" << comb_cfl << "). Setting dt to " << dt_new << std::endl;
    }

    // Don't let the timestep grow by more than m_dt_change_max per step
    // unless the previous time step was unduly shrunk to match m_plot_per_exact
    Real allowed_change_factor = m_dt_change_max;
    if ((m_dt > Real(0.0)) && !(m_plot_per_exact > 0 && m_last_plt == m_nstep && m_nstep > 0))
    {
        dt_new = amrex::min(dt_new, allowed_change_factor * m_prev_dt);
    }
    else if ((m_dt > Real(0.0)) && (m_plot_per_exact > 0 && m_last_plt == m_nstep && m_nstep > 0))
    {
        dt_new = amrex::min(dt_new, allowed_change_factor * amrex::max(m_prev_dt, m_prev_prev_dt));
    }

    // Don't overshoot specified plot times, but avoid tiny dt
    if (m_plot_per_exact > Real(0.0) &&
        (std::trunc((m_cur_time + dt_new + eps) / m_plot_per_exact) > std::trunc((m_cur_time + eps) / m_plot_per_exact)))
    {
        Real dt_plot = std::trunc((m_cur_time + dt_new) / m_plot_per_exact) * m_plot_per_exact - m_cur_time;
        // Enforce a minimum dt (m_dt_min if set, else 10x machine epsilon)
        Real min_dt = (m_dt_min > 0.0) ? m_dt_min : 10.0 * eps;
        if (dt_plot < min_dt) {
            // amrex::Print() << "WARNING: dt_new to match plot_per_exact would be too small (" << dt_plot << "). Using min_dt = " << min_dt << std::endl;
            if (dt_plot > 10.0 * eps) {
                dt_new = dt_plot;
            }
        } else {
            dt_new = dt_plot;
        }
    }

    // Don't overshoot the final time if not running to steady state
    if (!m_steady_state && m_stop_time > Real(0.0))
    {
        if (m_cur_time + dt_new > m_stop_time)
        {
            dt_new = m_stop_time - m_cur_time;
        }
    }

    // // For cryo runs, avoid pathological tiny positive timesteps during the main
    // // evolution. Allow smaller dt only when we are within the final-time window.
    // if (m_sim_cryo && m_dt_min > Real(0.0) && std::isfinite(dt_new) && dt_new > Real(0.0)
    //     && dt_new < m_dt_min && dt_new < tiny_dt_threshold)
    // {
    //     bool const near_final_time =
    //         (!m_steady_state && m_stop_time > Real(0.0) && (m_cur_time + m_dt_min > m_stop_time));

    //     if (!near_final_time)
    //     {
    //         amrex::Print() << "WARNING: dt_new=" << dt_new
    //                        << " is below dt_min=" << m_dt_min
    //                        << "; using dt_min instead." << std::endl;
    //         dt_new = m_dt_min;
    //     }
    // }

    // // Make sure the timestep remains finite and positive.
    // // Without this guard, a negative/invalid dt can make m_cur_time go backward.
    // if (!std::isfinite(dt_new) || dt_new <= Real(0.0))
    // {
    //     Real fallback_dt = Real(0.0);
    //     if (m_prev_dt > Real(0.0) && std::isfinite(m_prev_dt)) {
    //         fallback_dt = Real(0.5) * m_prev_dt;
    //     } else if (m_dt > Real(0.0) && std::isfinite(m_dt)) {
    //         fallback_dt = Real(0.5) * m_dt;
    //     }

    //     Real min_dt = (m_dt_min > Real(0.0)) ? m_dt_min : Real(10.0) * eps;
    //     fallback_dt = amrex::max(fallback_dt, min_dt);

    //     amrex::Print() << "WARNING: invalid dt_new=" << dt_new
    //                    << " (comb_cfl=" << comb_cfl
    //                    << ", conv_cfl=" << conv_cfl
    //                    << ", diff_cfl=" << diff_cfl
    //                    << ", forc_cfl=" << forc_cfl
    //                    << "). Using fallback dt=" << fallback_dt << std::endl;
    //     dt_new = fallback_dt;
    // }

    // If using fixed time step, check CFL condition and give warning if not satisfied
    if (m_fixed_dt > Real(0.0))
    {
        if (dt_new < m_fixed_dt)
        {
            amrex::Print() << "WARNING: fixed_dt does not satisfy CFL condition: \n"
                           << "max dt by CFL     : " << dt_new << "\n"
                           << "fixed dt specified: " << m_fixed_dt << std::endl;
        }
        m_dt = m_fixed_dt;
    }
    else
    {
        m_dt = dt_new;
    }

    if (!std::isfinite(m_dt) || m_dt <= Real(0.0)) {
        amrex::Abort("ComputeDt produced non-finite or non-positive m_dt");
    }
}
