#include <incflo.H>

using namespace amrex;

void incflo::compute_cp (int lev, MFIter& mfi, FArrayBox& cp) const
{
    Box const& bx = cp.box();
    Array4<Real> const& cp_a = cp.array();
    bool const conservative_temperature = !m_iconserv_temperature.empty() && m_iconserv_temperature[0] == 1;

    if (!conservative_temperature) {
        Real l_cp = m_cp;
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cp_a(i,j,k) = l_cp;
        });
        return;
    }

#ifdef INCFLO_SIM_CRYO
    auto const& ct = m_leveldata[lev]->cell_type.const_array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (ct(i,j,k) == -2) {
            cp_a(i,j,k) = cp_tcp;
        } else if (ct(i,j,k) == -3) {
            cp_a(i,j,k) = cp_plu;
        } else if (ct(i,j,k) == -4) {
            cp_a(i,j,k) = cp_sap;
        } else if (ct(i,j,k) == -5) {
            cp_a(i,j,k) = cp_dia;
        } else if (ct(i,j,k) == -6) {
            cp_a(i,j,k) = cp_plu;
        } else if (ct(i,j,k) == -1) {
            cp_a(i,j,k) = cp_eth;
        } else if (ct(i,j,k) >= 0) {
            cp_a(i,j,k) = cp_sam;
        } else {
            cp_a(i,j,k) = m_cp;
        }
    });
#endif
}
