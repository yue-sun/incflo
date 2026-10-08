#include <AMReX_ParmParse.H>
#include <AMReX_EB2.H>

#include <algorithm>
#include <incflo.H>

using namespace amrex;

void incflo::MakeEBGeometry()
{
   /******************************************************************************
   * incflo.geometry=<string> specifies the EB geometry. <string> can be one of    *
   * box, cylinder, annulus, sphere, spherecube, twocylinders
   ******************************************************************************/

    ParmParse pp("incflo");

    std::string geom_type;
    std::string csg_file;
    pp.query("geometry", geom_type);
    pp.query("geometry_filename", csg_file);
    amrex::Print() << "incflo.geometry_filename: " << csg_file;

#ifndef CSG_EB
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( csg_file.empty(), "CSG Geometry defined in input deck but solver not built with CSG support!");
#endif

    // EB on level 0 only: see m_eb_level0_only in incflo.H.  Only the cylinder
    // builder takes the level-0 geometry so far.
    pp.query("eb_level0_only", m_eb_level0_only);
    pp.query("eb_refine_margin", m_eb_refine_margin);
    if (m_eb_level0_only && geom_type != "cylinder") {
        amrex::Abort("incflo.eb_level0_only is only implemented for incflo.geometry = cylinder");
    }
    if (m_eb_level0_only && m_eb_refine_margin < 4) {
        amrex::Abort("incflo.eb_refine_margin must be >= 4 level-0 cells (error buffer + blocking factor)");
    }

   /******************************************************************************
   *                                                                            *
   *  CONSTRUCT EB                                                              *
   *                                                                            *
   ******************************************************************************/

    if(geom_type == "cylinder")
    {
    amrex::Print() << "\n Building cylinder geometry." << std::endl;
        make_eb_cylinder();
    }
    else if(geom_type == "box")
    {
        amrex::Print() << "\n Building box geometry." << std::endl;
        make_eb_box();
    }
#if (AMREX_SPACEDIM == 3)
    else if(geom_type == "twocylinders")
    {
    amrex::Print() << "\n Building twocylinders geometry." << std::endl;
        make_eb_twocylinders();
    }
    else if(geom_type == "spherecube")
    {
    amrex::Print() << "\n Building spherecube geometry." << std::endl;
        make_eb_spherecube();
    }
    else if(geom_type == "tuscan")
    {
    amrex::Print() << "\n Building tuscan geometry." << std::endl;
        make_eb_tuscan();
    }
#endif
    else if(geom_type == "annulus")
    {
    amrex::Print() << "\n Building annulus geometry." << std::endl;
        make_eb_annulus();
    }
    else if(geom_type == "sphere")
    {
    amrex::Print() << "\n Building sphere geometry." << std::endl;
        make_eb_sphere();
    }
    else if(geom_type == "jcap")
    {
    amrex::Print() << "\n Building JCAP geometry." << std::endl;
        make_eb_cyl_tuscan();
    }
    else if(geom_type == "chkptfile")
    {
       make_eb_chkptfile();
    }
#ifdef CSG_EB
    else if(!csg_file.empty()) {
      amrex::Print() << "\n Building geometry from .csg file:  " << csg_file << std::endl;
      make_eb_csg(csg_file);
    }
#endif
    else
    {
    amrex::Print() << "\n No EB geometry declared in inputs => "
                   << " Will build all regular geometry." << std::endl;
        make_eb_regular();
    }

    // Levels >= 1 get an all-regular index space (free: AMReX does not test any
    // boxes for AllRegularIF).  It goes on top of the stack, so AMReX's
    // implicit TopIndexSpace() users (FillPatchTwoLevels' coarse patch) see
    // regular coarse data -- true wherever a level >= 1 can be, given the
    // keep-out enforced in ErrorEst and make_eb_factory.
    if (m_eb_level0_only) {
        m_eb_wall_is = &amrex::EB2::IndexSpace::top();
        m_eb_fine_is = m_eb_wall_is;
        if (max_level > 0) {
            make_eb_regular();
            m_eb_fine_is = &amrex::EB2::IndexSpace::top();
        }
        amrex::Print() << " EB on level 0 only; levels 1-" << max_level
                       << " all regular, kept " << m_eb_refine_margin
                       << " level-0 cells from the wall." << std::endl;
    }
    amrex::Print() << "Done making the EB geometry index space.\n" << std::endl;

    if (m_write_geom_chk) {
       const auto& is = m_eb_level0_only ? *m_eb_wall_is : amrex::EB2::IndexSpace::top();
       const auto& eb_level = is.getLevel(m_eb_level0_only ? geom[0] : geom.back());
       eb_level.write_to_chkpt_file("geom_chk", amrex::EB2::ExtendDomainFace(), amrex::EB2::max_grid_size);
    }
}

std::unique_ptr<EBFArrayBoxFactory>
incflo::make_eb_factory (int lev, BoxArray const& ba, DistributionMapping const& dm) const
{
    Vector<int> const ng{nghost_eb_basic(), nghost_eb_volume(), nghost_eb_full()};
    if (!m_eb_level0_only) {
        return makeEBFabFactory(geom[lev], ba, dm, ng, EBSupport::full);
    }
    if (lev > 0) { eb_check_keepout(lev, ba); }
    return makeEBFabFactory((lev == 0) ? m_eb_wall_is : m_eb_fine_is, geom[lev], ba, dm, ng,
                            EBSupport::full);
}

// m_eb_keepout on the level-0 grids: 2 = cut or covered cell, 1 = regular but
// within m_eb_refine_margin cells of one, 0 = free to refine.  Level 0 covers
// the domain, so this sees the whole wall; one ghost cell for the dilation.
void incflo::eb_build_keepout ()
{
    BL_PROFILE("incflo::eb_build_keepout()");
    auto const& flags = EBFactory(0).getMultiEBCellFlagFab();
    m_eb_keepout.define(grids[0], dmap[0], 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(m_eb_keepout); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.fabbox();                   // flags have nghost_eb_basic() >= 1 ghosts
        auto const& fl = flags.const_array(mfi);
        auto const& ko = m_eb_keepout.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            ko(i,j,k) = fl(i,j,k).isRegular() ? 0 : 2;
        });
    }
    iMultiFab prev(grids[0], dmap[0], 1, 1);
    for (int pass = 0; pass < m_eb_refine_margin; ++pass) {
        m_eb_keepout.FillBoundary(geom[0].periodicity());
        iMultiFab::Copy(prev, m_eb_keepout, 0, 0, 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(m_eb_keepout, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.tilebox();
            auto const& p  = prev.const_array(mfi);
            auto const& ko = m_eb_keepout.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (ko(i,j,k) == 0) {
                    for (int kk = -1; kk <= 1; ++kk) {
                    for (int jj = -1; jj <= 1; ++jj) {
                    for (int ii = -1; ii <= 1; ++ii) {
                        if (p(i+ii,j+jj,k+kk) > 0) { ko(i,j,k) = 1; }
                    }}}
                }
            });
        }
    }
    m_eb_keepout.FillBoundary(geom[0].periodicity());
}

// Called at the end of ErrorEst: no level-1 grid may be created near the wall.
// The margin covers what comes after ErrorEst (n_error_buf, blocking-factor
// rounding); eb_check_keepout catches anything that still gets through.
void incflo::eb_clear_keepout_tags (int levc, TagBoxArray& tags)
{
    if (!m_eb_level0_only || levc != 0) { return; }
    if (m_eb_keepout.empty() || m_eb_keepout.boxArray() != grids[0] ||
        m_eb_keepout.DistributionMap() != dmap[0]) {
        eb_build_keepout();
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.tilebox();
        auto const& ko  = m_eb_keepout.const_array(mfi);
        auto const& tag = tags.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (ko(i,j,k) > 0) { tag(i,j,k) = TagBox::CLEAR; }
        });
    }
}

// A level-1 BoxArray (grown by the 2 level-0 cells its ghost cells can reach)
// must not overlap a cut or covered level-0 cell: level 1 has no EB.
void incflo::eb_check_keepout (int lev, BoxArray const& ba) const
{
    if (!m_eb_level0_only || lev != 1 || m_eb_keepout.empty()) { return; }
    BoxArray cba = amrex::coarsen(ba, ref_ratio[0]);
    cba.grow(2);
    iMultiFab hit(cba, DistributionMapping(cba), 1, 0);
    hit.setVal(0);
    hit.ParallelCopy(m_eb_keepout, 0, 0, 1, IntVect(0), IntVect(0), geom[0].periodicity());
    if (hit.max(0) >= 2) {
        amrex::Abort("incflo.eb_level0_only: a level-1 grid reaches the EB wall, where levels >= 1 "
                     "have no EB geometry.  Increase incflo.eb_refine_margin.");
    }
}
