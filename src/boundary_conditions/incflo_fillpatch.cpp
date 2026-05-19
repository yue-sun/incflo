#include <incflo.H>
#include <prob_bc.H>
#include <AMReX_FillPatchUtil.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBInterpolater.H>
#endif

using namespace amrex;

void incflo::fillpatch_velocity (int lev, Real time, MultiFab& vel, int ng)
{
    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloVelFill> > physbc
            (geom[lev], get_velocity_bcrec(),
             IncfloVelFill{m_probtype, m_bc_velocity});
        FillPatchSingleLevel(vel, IntVect(ng), time,
                             {&(m_leveldata[lev]->velocity_o),
                              &(m_leveldata[lev]->velocity)},
                             {m_t_old[lev], m_t_new[lev]}, 0, 0, AMREX_SPACEDIM, geom[lev],
                             physbc, 0);
    } else {
        const auto& bcrec = get_velocity_bcrec();
        PhysBCFunct<GpuBndryFuncFab<IncfloVelFill> > cphysbc
            (geom[lev-1], bcrec, IncfloVelFill{m_probtype, m_bc_velocity});
        PhysBCFunct<GpuBndryFuncFab<IncfloVelFill> > fphysbc
            (geom[lev], bcrec, IncfloVelFill{m_probtype, m_bc_velocity});
#ifdef AMREX_USE_EB
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
        Interpolater* mapper = &cell_cons_interp;
#endif
        FillPatchTwoLevels(vel, IntVect(ng), time,
                           {&(m_leveldata[lev-1]->velocity_o),
                            &(m_leveldata[lev-1]->velocity)},
                           {m_t_old[lev-1], m_t_new[lev-1]},
                           {&(m_leveldata[lev]->velocity_o),
                            &(m_leveldata[lev]->velocity)},
                           {m_t_old[lev], m_t_new[lev]},
                           0, 0, AMREX_SPACEDIM, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }
}

void incflo::fillpatch_density (int lev, Real time, MultiFab& density, int ng)
{
    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > physbc(geom[lev], get_density_bcrec(),
                                                            IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
        FillPatchSingleLevel(density, IntVect(ng), time,
                             {&(m_leveldata[lev]->density_o),
                              &(m_leveldata[lev]->density)},
                             {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev],
                             physbc, 0);
    } else {
        const auto& bcrec = get_density_bcrec();
        PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > cphysbc
            (geom[lev-1], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
        PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > fphysbc
            (geom[lev], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
#ifdef AMREX_USE_EB
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
        Interpolater* mapper = &cell_cons_interp;
#endif
        FillPatchTwoLevels(density, IntVect(ng), time,
                           {&(m_leveldata[lev-1]->density_o),
                            &(m_leveldata[lev-1]->density)},
                           {m_t_old[lev-1], m_t_new[lev-1]},
                           {&(m_leveldata[lev]->density_o),
                            &(m_leveldata[lev]->density)},
                           {m_t_old[lev], m_t_new[lev]},
                           0, 0, 1, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }
}

void incflo::fillpatch_cell_type (int lev, Real time, iMultiFab& cell_type, int ng)
{
    MultiFab cell_type_real(cell_type.boxArray(), cell_type.DistributionMap(), 1, ng);
    MultiFab cell_type_old_real(m_leveldata[lev]->cell_type_o.boxArray(),
                                m_leveldata[lev]->cell_type_o.DistributionMap(),
                                1, m_leveldata[lev]->cell_type_o.nGrow());
    MultiFab cell_type_new_real(m_leveldata[lev]->cell_type.boxArray(),
                                m_leveldata[lev]->cell_type.DistributionMap(),
                                1, m_leveldata[lev]->cell_type.nGrow());

    for (MFIter mfi(cell_type_old_real, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.growntilebox();
        Array4<Real> const& old_arr = cell_type_old_real.array(mfi);
        Array4<Real> const& new_arr = cell_type_new_real.array(mfi);
        Array4<int const> const& old_src = m_leveldata[lev]->cell_type_o.const_array(mfi);
        Array4<int const> const& new_src = m_leveldata[lev]->cell_type.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            old_arr(i,j,k) = static_cast<Real>(old_src(i,j,k));
            new_arr(i,j,k) = static_cast<Real>(new_src(i,j,k));
        });
    }

    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > physbc(geom[lev], get_density_bcrec(),
                                                            IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
        FillPatchSingleLevel(cell_type_real, IntVect(ng), time,
                             {&cell_type_old_real, &cell_type_new_real},
                             {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev],
                             physbc, 0);
    } else {
        MultiFab cell_type_old_coarse_real(m_leveldata[lev-1]->cell_type_o.boxArray(),
                                           m_leveldata[lev-1]->cell_type_o.DistributionMap(),
                                           1, m_leveldata[lev-1]->cell_type_o.nGrow());
        MultiFab cell_type_new_coarse_real(m_leveldata[lev-1]->cell_type.boxArray(),
                                           m_leveldata[lev-1]->cell_type.DistributionMap(),
                                           1, m_leveldata[lev-1]->cell_type.nGrow());

        for (MFIter mfi(cell_type_old_coarse_real, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.growntilebox();
            Array4<Real> const& old_arr = cell_type_old_coarse_real.array(mfi);
            Array4<Real> const& new_arr = cell_type_new_coarse_real.array(mfi);
            Array4<int const> const& old_src = m_leveldata[lev-1]->cell_type_o.const_array(mfi);
            Array4<int const> const& new_src = m_leveldata[lev-1]->cell_type.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                old_arr(i,j,k) = static_cast<Real>(old_src(i,j,k));
                new_arr(i,j,k) = static_cast<Real>(new_src(i,j,k));
            });
        }

        const auto& bcrec = get_density_bcrec();
        PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > cphysbc
            (geom[lev-1], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
        PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > fphysbc
            (geom[lev], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
#ifdef AMREX_USE_EB
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
        Interpolater* mapper = &cell_cons_interp;
#endif
        FillPatchTwoLevels(cell_type_real, IntVect(ng), time,
                           {&cell_type_old_coarse_real, &cell_type_new_coarse_real},
                           {m_t_old[lev-1], m_t_new[lev-1]},
                           {&cell_type_old_real, &cell_type_new_real},
                           {m_t_old[lev], m_t_new[lev]},
                           0, 0, 1, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }

    for (MFIter mfi(cell_type_real, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.tilebox();
        Array4<Real const> const& src = cell_type_real.const_array(mfi);
        Array4<int> const& dst = cell_type.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const v = src(i,j,k);
            dst(i,j,k) = static_cast<int>(v + (v >= Real(0.0) ? Real(0.5) : Real(-0.5)));
        });
    }
}

void incflo::fillpatch_tracer (int lev, Real time, MultiFab& tracer, int ng)
{
    if (m_ntrac <= 0) return;
    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > physbc
            (geom[lev], get_tracer_bcrec(), IncfloTracFill{m_probtype, m_ntrac, m_bc_tracer_d, m_bc_velocity});
        FillPatchSingleLevel(tracer, IntVect(ng), time,
                             {&(m_leveldata[lev]->tracer_o),
                              &(m_leveldata[lev]->tracer)},
                             {m_t_old[lev], m_t_new[lev]}, 0, 0, m_ntrac, geom[lev],
                             physbc, 0);
    } else {
        const auto& bcrec = get_tracer_bcrec();
        PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > cphysbc
            (geom[lev-1], bcrec, IncfloTracFill{m_probtype, m_ntrac, m_bc_tracer_d, m_bc_velocity});
        PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > fphysbc
            (geom[lev], bcrec, IncfloTracFill{m_probtype, m_ntrac, m_bc_tracer_d, m_bc_velocity});
#ifdef AMREX_USE_EB
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
        Interpolater* mapper = &cell_cons_interp;
#endif
        FillPatchTwoLevels(tracer, IntVect(ng), time,
                           {&(m_leveldata[lev-1]->tracer_o),
                            &(m_leveldata[lev-1]->tracer)},
                           {m_t_old[lev-1], m_t_new[lev-1]},
                           {&(m_leveldata[lev]->tracer_o),
                            &(m_leveldata[lev]->tracer)},
                           {m_t_old[lev], m_t_new[lev]},
                           0, 0, m_ntrac, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }
}

void incflo::fillpatch_temperature (int lev, Real time, MultiFab& temperature, int ng)
{
    if (!m_use_temperature) return;

    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloTempFill> > physbc
            (geom[lev], get_temperature_bcrec(), IncfloTempFill{m_probtype, m_bc_temperature, m_bc_velocity});
        FillPatchSingleLevel(temperature, IntVect(ng), time,
                             {&(m_leveldata[lev]->temperature_o),
                              &(m_leveldata[lev]->temperature)},
                             {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev],
                             physbc, 0);
    } else {
        const auto& bcrec = get_temperature_bcrec();
        PhysBCFunct<GpuBndryFuncFab<IncfloTempFill> > cphysbc
            (geom[lev-1], bcrec, IncfloTempFill{m_probtype, m_bc_temperature, m_bc_velocity});
        PhysBCFunct<GpuBndryFuncFab<IncfloTempFill> > fphysbc
            (geom[lev], bcrec, IncfloTempFill{m_probtype, m_bc_temperature, m_bc_velocity});
#ifdef AMREX_USE_EB
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
        Interpolater* mapper = &cell_cons_interp;
#endif
        FillPatchTwoLevels(temperature, IntVect(ng), time,
                           {&(m_leveldata[lev-1]->temperature_o),
                            &(m_leveldata[lev-1]->temperature)},
                           {m_t_old[lev-1], m_t_new[lev-1]},
                           {&(m_leveldata[lev]->temperature_o),
                            &(m_leveldata[lev]->temperature)},
                           {m_t_old[lev], m_t_new[lev]},
                           0, 0, 1, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }
}

void incflo::fillpatch_gradp (int lev, Real time, MultiFab& gp, int ng)
{
    if (lev == 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > physbc
            (geom[lev], get_force_bcrec(), IncfloForFill{m_probtype});
        FillPatchSingleLevel(gp, IntVect(ng), time,
                             {&(m_leveldata[lev]->gp)}, {time},
                             0, 0, AMREX_SPACEDIM, geom[lev], physbc, 0);
    } else {
        const auto& bcrec = get_force_bcrec();
        PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > cphysbc
            (geom[lev-1], bcrec, IncfloForFill{m_probtype});
        PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > fphysbc
            (geom[lev], bcrec, IncfloForFill{m_probtype});
#ifdef AMREX_USE_EB
        Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
            (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
        Interpolater* mapper = &cell_cons_interp;
#endif
        FillPatchTwoLevels(gp, IntVect(ng), time,
                           {&(m_leveldata[lev-1]->gp)}, {time},
                           {&(m_leveldata[lev]->gp)}, {time},
                           0, 0, AMREX_SPACEDIM, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }
}

void incflo::fillpatch_force (Real time, Vector<MultiFab*> const& force, int ng)
{
    const int ncomp = force[0]->nComp();
    const auto& bcrec = get_force_bcrec();
    int lev = 0;
    {
        PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > physbc
            (geom[lev], bcrec, IncfloForFill{m_probtype});
        FillPatchSingleLevel(*force[lev], IntVect(ng), time,
                             {force[lev]}, {time},
                             0, 0, ncomp, geom[lev],
                             physbc, 0);
    }
    for (lev = 1; lev <= finest_level; ++lev)
    {
        PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > cphysbc
            (geom[lev-1], bcrec, IncfloForFill{m_probtype});
        PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > fphysbc
            (geom[lev  ], bcrec, IncfloForFill{m_probtype});
        Interpolater* mapper = &pc_interp;
        FillPatchTwoLevels(*force[lev], IntVect(ng), time,
                           {force[lev-1]}, {time},
                           {force[lev  ]}, {time},
                           0, 0, ncomp, geom[lev-1], geom[lev],
                           cphysbc, 0, fphysbc, 0,
                           refRatio(lev-1), mapper, bcrec, 0);
    }
}

void incflo::fillcoarsepatch_velocity (int lev, Real time, MultiFab& vel, int ng)
{
    const auto& bcrec = get_velocity_bcrec();
    PhysBCFunct<GpuBndryFuncFab<IncfloVelFill> > cphysbc
        (geom[lev-1], bcrec, IncfloVelFill{m_probtype, m_bc_velocity});
    PhysBCFunct<GpuBndryFuncFab<IncfloVelFill> > fphysbc
        (geom[lev], bcrec, IncfloVelFill{m_probtype, m_bc_velocity});
#ifdef AMREX_USE_EB
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
    Interpolater* mapper = &cell_cons_interp;
#endif
    amrex::InterpFromCoarseLevel(vel, IntVect(ng), time,
                                 m_leveldata[lev-1]->velocity, 0, 0, AMREX_SPACEDIM,
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcrec, 0);
}

void incflo::fillcoarsepatch_density (int lev, Real time, MultiFab& density, int ng)
{
    const auto& bcrec = get_density_bcrec();
    PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > cphysbc
        (geom[lev-1], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
    PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > fphysbc
        (geom[lev], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
#ifdef AMREX_USE_EB
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
    Interpolater* mapper = &cell_cons_interp;
#endif
    amrex::InterpFromCoarseLevel(density, IntVect(ng), time,
                                 m_leveldata[lev-1]->density, 0, 0, 1,
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcrec, 0);
}

void incflo::fillcoarsepatch_cell_type (int lev, Real time, iMultiFab& cell_type, int ng)
{
    MultiFab cell_type_real(cell_type.boxArray(), cell_type.DistributionMap(), 1, ng);
    MultiFab cell_type_coarse_real(m_leveldata[lev-1]->cell_type.boxArray(),
                                   m_leveldata[lev-1]->cell_type.DistributionMap(),
                                   1, m_leveldata[lev-1]->cell_type.nGrow());

    for (MFIter mfi(cell_type_coarse_real, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.growntilebox();
        Array4<Real> const& dst = cell_type_coarse_real.array(mfi);
        Array4<int const> const& src = m_leveldata[lev-1]->cell_type.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            dst(i,j,k) = static_cast<Real>(src(i,j,k));
        });
    }

    const auto& bcrec = get_density_bcrec();
    PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > cphysbc
        (geom[lev-1], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
    PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > fphysbc
        (geom[lev], bcrec, IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
#ifdef AMREX_USE_EB
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
    Interpolater* mapper = &cell_cons_interp;
#endif
    amrex::InterpFromCoarseLevel(cell_type_real, IntVect(ng), time,
                                 cell_type_coarse_real, 0, 0, 1,
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcrec, 0);

    for (MFIter mfi(cell_type_real, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.tilebox();
        Array4<Real const> const& src = cell_type_real.const_array(mfi);
        Array4<int> const& dst = cell_type.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const v = src(i,j,k);
            dst(i,j,k) = static_cast<int>(v + (v >= Real(0.0) ? Real(0.5) : Real(-0.5)));
        });
    }
}

void incflo::fillcoarsepatch_tracer (int lev, Real time, MultiFab& tracer, int ng)
{
    if (m_ntrac <= 0) return;

    const auto& bcrec = get_tracer_bcrec();
    PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > cphysbc
        (geom[lev-1], bcrec, IncfloTracFill{m_probtype, m_ntrac, m_bc_tracer_d, m_bc_velocity});
    PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > fphysbc
        (geom[lev], bcrec, IncfloTracFill{m_probtype, m_ntrac, m_bc_tracer_d, m_bc_velocity});
#ifdef AMREX_USE_EB
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
    Interpolater* mapper = &cell_cons_interp;
#endif
    amrex::InterpFromCoarseLevel(tracer, IntVect(ng), time,
                                 m_leveldata[lev-1]->tracer, 0, 0, m_ntrac,
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcrec, 0);
}

void incflo::fillcoarsepatch_temperature (int lev, Real time, MultiFab& temperature, int ng)
{
    if (m_use_temperature) return;

    const auto& bcrec = get_temperature_bcrec();
    PhysBCFunct<GpuBndryFuncFab<IncfloTempFill> > cphysbc
        (geom[lev-1], bcrec, IncfloTempFill{m_probtype, m_bc_temperature, m_bc_velocity});
    PhysBCFunct<GpuBndryFuncFab<IncfloTempFill> > fphysbc
        (geom[lev], bcrec, IncfloTempFill{m_probtype, m_bc_temperature, m_bc_velocity});
#ifdef AMREX_USE_EB
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
    Interpolater* mapper = &cell_cons_interp;
#endif
    amrex::InterpFromCoarseLevel(temperature, IntVect(ng), time,
                                 m_leveldata[lev-1]->temperature, 0, 0, 1,
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcrec, 0);
}

void incflo::fillcoarsepatch_gradp (int lev, Real time, MultiFab& gp, int ng)
{
    const auto& bcrec = get_force_bcrec();
    PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > cphysbc
        (geom[lev-1], bcrec, IncfloForFill{m_probtype});
    PhysBCFunct<GpuBndryFuncFab<IncfloForFill> > fphysbc
        (geom[lev], bcrec, IncfloForFill{m_probtype});
#ifdef AMREX_USE_EB
    Interpolater* mapper = (EBFactory(0).isAllRegular()) ?
        (Interpolater*)(&cell_cons_interp) : (Interpolater*)(&eb_cell_cons_interp);
#else
    Interpolater* mapper = &cell_cons_interp;
#endif
    amrex::InterpFromCoarseLevel(gp, IntVect(ng), time,
                                 m_leveldata[lev-1]->gp, 0, 0, AMREX_SPACEDIM,
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcrec, 0);
}
