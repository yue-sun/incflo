#include <AMReX_BC_TYPES.H>
#include <incflo.H>
#include <hydro_mol.H>

using namespace amrex;

void
average_ccvel_to_mac (const Array<MultiFab*,AMREX_SPACEDIM>& fc, const MultiFab& cc)
{
        AMREX_ASSERT(cc.nComp() == AMREX_SPACEDIM);
        AMREX_ASSERT(cc.nGrow() >= 1);
        AMREX_ASSERT(fc[0]->nComp() == 1);
#if (AMREX_SPACEDIM >= 2)
        AMREX_ASSERT(fc[1]->nComp() == 1);
#endif
#if (AMREX_SPACEDIM == 3)
        AMREX_ASSERT(fc[2]->nComp() == 1);
#endif

        int ncomp = AMREX_SPACEDIM;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion() && cc.isFusingCandidate()) {
            auto const& ccma = cc.const_arrays();
            AMREX_D_TERM(auto const& fxma = fc[0]->arrays();,
                         auto const& fyma = fc[1]->arrays();,
                         auto const& fzma = fc[2]->arrays(););
            MultiFab foo(amrex::convert(cc.boxArray(),IntVect(1)), cc.DistributionMap(), 1, 0,
                         MFInfo().SetAlloc(false));
            IntVect ng = -cc.nGrowVect();
            ParallelFor(foo, IntVect(0), ncomp,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
            {
                Box ccbx(ccma[box_no]);
                ccbx.grow(ng);
                AMREX_D_TERM(Box const& xbx = amrex::surroundingNodes(ccbx,0);,
                             Box const& ybx = amrex::surroundingNodes(ccbx,1);,
                             Box const& zbx = amrex::surroundingNodes(ccbx,2););
                if (xbx.contains(i,j,k) and n == 0) {
                    fxma[box_no](i,j,k) = Real(0.5)*(ccma[box_no](i-1,j,k,n) + ccma[box_no](i,j,k,n));
                }
                if (ybx.contains(i,j,k) and n == 1) {
                    fyma[box_no](i,j,k) = Real(0.5)*(ccma[box_no](i,j-1,k,n) + ccma[box_no](i,j,k,n));
                }
#if (AMREX_SPACEDIM == 3)
                if (zbx.contains(i,j,k) and n == 2) {
                    fzma[box_no](i,j,k) = Real(0.5)*(ccma[box_no](i,j,k-1,n) + ccma[box_no](i,j,k,n));
                }
#endif
            });
            Gpu::streamSynchronize();
        } else
#endif
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                AMREX_D_TERM(const Box& xbx = mfi.nodaltilebox(0);,
                             const Box& ybx = mfi.nodaltilebox(1);,
                             const Box& zbx = mfi.nodaltilebox(2););
                const auto& index_bounds = amrex::getIndexBounds(AMREX_D_DECL(xbx,ybx,zbx));

                AMREX_D_TERM(Array4<Real> const& fxarr = fc[0]->array(mfi);,
                             Array4<Real> const& fyarr = fc[1]->array(mfi);,
                             Array4<Real> const& fzarr = fc[2]->array(mfi););
                Array4<Real const> const& ccarr = cc.const_array(mfi);

#if (AMREX_SPACEDIM == 2)
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(index_bounds, ncomp, i, j, k, n,
                {
                   if (xbx.contains(i,j,k) and n == 0) {
                       fxarr(i,j,k) = Real(0.5)*(ccarr(i-1,j,k,n) + ccarr(i,j,k,n));
                   }
                   if (ybx.contains(i,j,k) and n == 1) {
                       fyarr(i,j,k) = Real(0.5)*(ccarr(i,j-1,k,n) + ccarr(i,j,k,n));
                   }
                });
#else
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(index_bounds, ncomp, i, j, k, n,
                {
                   if (xbx.contains(i,j,k) and n == 0) {
                       fxarr(i,j,k) = Real(0.5)*(ccarr(i-1,j,k,n) + ccarr(i,j,k,n));
                   }
                   if (ybx.contains(i,j,k) and n == 1) {
                       fyarr(i,j,k) = Real(0.5)*(ccarr(i,j-1,k,n) + ccarr(i,j,k,n));
                   }
                   if (zbx.contains(i,j,k) and n == 2) {
                       fzarr(i,j,k) = Real(0.5)*(ccarr(i,j,k-1,n) + ccarr(i,j,k,n));
                   }
                });
#endif
            }
        }
}

void
average_mac_to_ccvel (const Array<MultiFab*,AMREX_SPACEDIM>& fc, MultiFab& cc)
{
        AMREX_ASSERT(cc.nComp() == AMREX_SPACEDIM);
        AMREX_ASSERT(fc[0]->nComp() == 1);
#if (AMREX_SPACEDIM >= 2)
        AMREX_ASSERT(fc[1]->nComp() == 1);
#endif
#if (AMREX_SPACEDIM == 3)
        AMREX_ASSERT(fc[2]->nComp() == 1);
#endif

        int ncomp = AMREX_SPACEDIM;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion() && cc.isFusingCandidate()) {
            auto const& ccma = cc.arrays();
            AMREX_D_TERM(auto const& fxma = fc[0]->arrays();,
                         auto const& fyma = fc[1]->arrays();,
                         auto const& fzma = fc[2]->arrays(););
            MultiFab foo(amrex::convert(cc.boxArray(),IntVect(1)), cc.DistributionMap(), 1, 0,
                         MFInfo().SetAlloc(false));
            ParallelFor(foo, IntVect(0), ncomp,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
            {
                Box ccbx(ccma[box_no]);
                AMREX_D_TERM(Box const& xbx = amrex::surroundingNodes(ccbx,0);,
                             Box const& ybx = amrex::surroundingNodes(ccbx,1);,
                             Box const& zbx = amrex::surroundingNodes(ccbx,2););
                if (xbx.contains(i,j,k) and n == 0) {
                    ccma[box_no](i,j,k,n) = Real(0.5) * (fxma[box_no](i,j,k) + fxma[box_no](i+1,j,k));
                }
                if (ybx.contains(i,j,k) and n == 1) {
                    ccma[box_no](i,j,k,n) = Real(0.5) * (fyma[box_no](i,j,k) + fyma[box_no](i,j+1,k));
                }
#if (AMREX_SPACEDIM == 3)
                if (zbx.contains(i,j,k) and n == 2) {
                    ccma[box_no](i,j,k,n) = Real(0.5) * (fzma[box_no](i,j,k) + fzma[box_no](i,j,k+1));
                }
#endif
            });
            Gpu::streamSynchronize();
        } else
#endif
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

                AMREX_D_TERM(Array4<Real> const& fxarr = fc[0]->array(mfi);,
                             Array4<Real> const& fyarr = fc[1]->array(mfi);,
                             Array4<Real> const& fzarr = fc[2]->array(mfi););
                Array4<Real> const& ccarr = cc.array(mfi);

#if (AMREX_SPACEDIM == 2)
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
                {
                   if (n == 0) {
                       ccarr(i,j,k,0) = Real(0.5) * (fxarr(i,j,k) + fxarr(i+1,j,k));
                   } else if (n == 1) {
                       ccarr(i,j,k,1) = Real(0.5) * (fyarr(i,j,k) + fyarr(i,j+1,k));
                   }
                });
#else
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
                {
                   if (n == 0) {
                       ccarr(i,j,k,0) = Real(0.5) * (fxarr(i,j,k) + fxarr(i+1,j,k));
                   } else if (n == 1) {
                       ccarr(i,j,k,1) = Real(0.5) * (fyarr(i,j,k) + fyarr(i,j+1,k));
                   } else if (n == 2) {
                       ccarr(i,j,k,2) = Real(0.5) * (fzarr(i,j,k) + fzarr(i,j,k+1));
                   }
                });
#endif
            }
        }
}

//
// Computes the following decomposition:
//
//    u + dt grad(phi) / ro = u*,     where div(u) = 0
//
// where u* is a non-div-free velocity field, stored
// by components in u, v, and w. The resulting div-free
// velocity field, u, overwrites the value of u* in u, v, and w.
//
// phi is an auxiliary function related to the pressure p by the relation:
//
//     new p  = phi
//
// except in the initial projection when
//
//     new p  = old p + phi     (nstep has its initial value -1)
//
// Note: scaling_factor equals dt except when called during initial projection, when it is 1.0
//
void incflo::ApplyCCProjection (Vector<MultiFab const*> density,
                                AMREX_D_DECL(Vector<MultiFab*> const& u_mac,
                                             Vector<MultiFab*> const& v_mac,
                                             Vector<MultiFab*> const& w_mac),
                                Real time, Real scaling_factor, bool incremental)
{
    BL_PROFILE("incflo::ApplyCCProjection");

    // If we have dropped the dt substantially for whatever reason,
    // use a different form of the approximate projection that
    // projects (U^*-U^n + dt Gp) rather than (U^* + dt Gp)

    bool proj_for_small_dt = (time > 0.0 && m_dt < 0.1 * m_prev_dt);

    // Add ( dt grad p /ro ) to u*
    if (!incremental)
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real>       const& u   = ld.velocity.array(mfi);
                Array4<Real const> const& gp  = ld.gp.const_array(mfi);
                Array4<Real const> const& rho = density[lev]->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real soverrho = scaling_factor / rho(i,j,k);
                    AMREX_D_TERM(u(i,j,k,0) += gp(i,j,k,0) * soverrho;,
                                 u(i,j,k,1) += gp(i,j,k,1) * soverrho;,
                                 u(i,j,k,2) += gp(i,j,k,2) * soverrho;);
                });
            }
        }
    }

    // Define "vel" to be U^* - U^n rather than U^*
    if (proj_for_small_dt || incremental)
    {
        for (int lev = 0; lev <= finest_level; ++lev) {
            MultiFab::Subtract(m_leveldata[lev]->velocity,
                               m_leveldata[lev]->velocity_o, 0, 0, AMREX_SPACEDIM, 0);
        }
    }

    auto bclo = get_projection_bc(Orientation::low);
    auto bchi = get_projection_bc(Orientation::high);

    Vector<MultiFab*> vel;
    for (int lev = 0; lev <= finest_level; ++lev) {
        vel.push_back(&(m_leveldata[lev]->velocity));
        vel[lev]->setBndry(0.0);
        if (!proj_for_small_dt && !incremental) {
            set_inflow_velocity(lev, time, *vel[lev], 1);
        }
    }

    // ***************************************************************************************
    // START OF MAC STUFF
    // ***************************************************************************************

    // This will hold (dt/rho) on faces
    Vector<Array<MultiFab,AMREX_SPACEDIM> > inv_rho(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev) {
        AMREX_D_TERM(
           inv_rho[lev][0].define(u_mac[lev]->boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));,
           inv_rho[lev][1].define(v_mac[lev]->boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));,
           inv_rho[lev][2].define(w_mac[lev]->boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev)));
        AMREX_D_TERM(
           inv_rho[lev][0].setVal(0.);,
           inv_rho[lev][1].setVal(0.);,
           inv_rho[lev][2].setVal(0.););
    }

    // Compute (1/rho) on faces
    for (int lev=0; lev <= finest_level; ++lev)
    {
#ifdef AMREX_USE_EB
        EB_interp_CellCentroid_to_FaceCentroid (*density[lev], GetArrOfPtrs(inv_rho[lev]),
                                                0, 0, 1, geom[lev], get_density_bcrec());
#else
        amrex::average_cellcenter_to_face(GetArrOfPtrs(inv_rho[lev]), *density[lev], geom[lev]);
#endif

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            inv_rho[lev][idim].invert(scaling_factor, 0);
        }
    }

    //
    // Initialize (or redefine the beta in) the MacProjector
    if (macproj->needInitialization())
    {
        LPInfo lp_info;
        lp_info.setMaxCoarseningLevel(m_mac_mg_max_coarsening_level);
#ifndef AMREX_USE_EB
        if (m_constant_density) {
            Vector<BoxArray> ba;
            Vector<DistributionMapping> dm;
            for (auto const& ir : inv_rho) {
                ba.push_back(ir[0].boxArray());
                dm.push_back(ir[0].DistributionMap());
            }
            macproj->initProjector(ba, dm, lp_info, scaling_factor/m_ro_0);
        } else
#endif
        {
            macproj->initProjector(lp_info, GetVecOfArrOfConstPtrs(inv_rho));
        }
        macproj->setDomainBC(bclo, bchi);
    } else {
#ifndef AMREX_USE_EB
        if (m_constant_density) {
            macproj->updateBeta(scaling_factor/m_ro_0);  // unnecessary unless m_ro_0 changes.
        } else
#endif
        {
            macproj->updateBeta(GetVecOfArrOfConstPtrs(inv_rho));
        }
    }

    Vector<MultiFab*> cc_phi(finest_level+1);
    Vector<MultiFab*> cc_gphi(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev )
    {
        cc_phi[lev]  = new MultiFab(grids[lev], dmap[lev], 1, 1, MFInfo(), Factory(lev));
        cc_gphi[lev] = new MultiFab(grids[lev], dmap[lev], AMREX_SPACEDIM, 0, MFInfo(), Factory(lev));
        cc_phi[lev]->setVal(0.);
        cc_gphi[lev]->setVal(0.);
    }

    Vector<Array<MultiFab*,AMREX_SPACEDIM> > mac_vec(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        AMREX_D_TERM(mac_vec[lev][0] = u_mac[lev];,
                     mac_vec[lev][1] = v_mac[lev];,
                     mac_vec[lev][2] = w_mac[lev];);
    }

    // Compute velocity on faces
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Predict normal velocity to faces -- note that the {u_mac, v_mac, w_mac}
        //    returned from this call are on face CENTROIDS
        vel[lev]->FillBoundary(geom[lev].periodicity());
#if 1
        MOL::ExtrapVelToFaces(*vel[lev],
                              AMREX_D_DECL(*u_mac[lev], *v_mac[lev], *w_mac[lev]),
                              geom[lev],
                              get_velocity_bcrec(), get_velocity_bcrec_device_ptr());

#else
        average_ccvel_to_mac(             mac_vec[lev],      *vel[lev]);
#endif
    }

    macproj->setUMAC(mac_vec);

    if (m_verbose > 2) amrex::Print() << "CC Projection:\n";
    //
    // Perform MAC projection:  - del dot (dt/rho) grad phi = div(U)
    //
    macproj->project(cc_phi,m_mac_mg_rtol,m_mac_mg_atol);

    //
    // After the projection we grab the dt/rho (grad phi) used in the projection
    //
    Vector<Array<MultiFab,AMREX_SPACEDIM> > m_fluxes;
    m_fluxes.resize(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
             m_fluxes[lev][idim].define(
                    amrex::convert(grids[lev], IntVect::TheDimensionVector(idim)),
                    dmap[lev], 1, 0, MFInfo(), Factory(lev));
        }
    }

    //
    // Note that "fluxes" comes back as MINUS (dt/rho) Gphi
    //
#ifdef AMREX_USE_EB
    macproj->getFluxes(amrex::GetVecOfArrOfPtrs(m_fluxes), cc_phi, MLMG::Location::FaceCentroid);
#else
    macproj->getFluxes(amrex::GetVecOfArrOfPtrs(m_fluxes), cc_phi, MLMG::Location::FaceCenter);
#endif

    for (int lev=0; lev <= finest_level; ++lev)
    {
#ifdef AMREX_USE_EB
        amrex::Abort("Haven't written mac_to_ccvel for EB");
#else
        average_mac_to_ccvel(GetArrOfPtrs(m_fluxes[lev]),*cc_gphi[lev]);
#endif
    }

    for(int lev = 0; lev <= finest_level; lev++)
    {
        auto& ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.gp,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& tbx = mfi.tilebox();
            Array4<Real> const& gp_cc = ld.gp.array(mfi);
            Array4<Real> const&  p_cc = ld.p_cc.array(mfi);
            Array4<Real const> const& gphi = cc_gphi[lev]->const_array(mfi);
            Array4<Real const> const&  phi = cc_phi[lev]->const_array(mfi);

            Array4<Real> const& u = ld.velocity.array(mfi);
            Array4<Real const> const& rho = density[lev]->const_array(mfi);

            Real r0 = m_ro_0;

            amrex::ParallelFor(tbx, [u,gphi,p_cc,phi,incremental] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                AMREX_D_TERM(u(i,j,k,0) += gphi(i,j,k,0);,
                             u(i,j,k,1) += gphi(i,j,k,1);,
                             u(i,j,k,2) += gphi(i,j,k,2););
                if (incremental)
                    p_cc (i,j,k) += phi(i,j,k);
                else
                    p_cc (i,j,k)  = phi(i,j,k);
            });

            if (incremental && m_constant_density) {
                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(gp_cc(i,j,k,0) -= gphi(i,j,k,0) * r0 / scaling_factor;,
                                 gp_cc(i,j,k,1) -= gphi(i,j,k,1) * r0 / scaling_factor;,
                                 gp_cc(i,j,k,2) -= gphi(i,j,k,2) * r0 / scaling_factor;);

                });
            } else if (incremental && !m_constant_density) {
                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(gp_cc(i,j,k,0) -= gphi(i,j,k,0) * rho(i,j,k) / scaling_factor;,
                                 gp_cc(i,j,k,1) -= gphi(i,j,k,1) * rho(i,j,k) / scaling_factor;,
                                 gp_cc(i,j,k,2) -= gphi(i,j,k,2) * rho(i,j,k) / scaling_factor;);
                });
            } else if (!incremental && m_constant_density) {
                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(gp_cc(i,j,k,0) = -gphi(i,j,k,0) * r0 / scaling_factor;,
                                 gp_cc(i,j,k,1) = -gphi(i,j,k,1) * r0 / scaling_factor;,
                                 gp_cc(i,j,k,2) = -gphi(i,j,k,2) * r0 / scaling_factor;);
                });
            } else if (!incremental && !m_constant_density) {
                amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(gp_cc(i,j,k,0) = -gphi(i,j,k,0) * rho(i,j,k) / scaling_factor;,
                                 gp_cc(i,j,k,1) = -gphi(i,j,k,1) * rho(i,j,k) / scaling_factor;,
                                 gp_cc(i,j,k,2) = -gphi(i,j,k,2) * rho(i,j,k) / scaling_factor;);
                });
            }
        }
        ld.gp.FillBoundary(geom[lev].periodicity());
        ld.p_cc.FillBoundary(geom[lev].periodicity());
    }

    // ***************************************************************************************
    // END OF MAC STUFF
    // ***************************************************************************************

    // Define "vel" to be U^{n+1} rather than (U^{n+1}-U^n)
    if (proj_for_small_dt || incremental)
    {
        for (int lev = 0; lev <= finest_level; ++lev) {
            MultiFab::Add(m_leveldata[lev]->velocity,
                          m_leveldata[lev]->velocity_o, 0, 0, AMREX_SPACEDIM, 0);
        }
    }

    for (int lev = finest_level-1; lev >= 0; --lev) {
#ifdef AMREX_USE_EB
        amrex::EB_average_down(m_leveldata[lev+1]->gp, m_leveldata[lev]->gp,
                               0, AMREX_SPACEDIM, refRatio(lev));
#else
        amrex::average_down(m_leveldata[lev+1]->gp, m_leveldata[lev]->gp,
                            0, AMREX_SPACEDIM, refRatio(lev));
#endif
    }
}
