#ifndef INCFLO_CRYO_INTERP_H_
#define INCFLO_CRYO_INTERP_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

namespace cryo_interp
{

    // -----------------------------------------------------------------------
    // Piecewise cubic evaluator  f(x) = c3·dx³ + c2·dx² + c1·dx + c0
    // where dx = x - left knot of the containing segment.
    // Layout: xs[N+1] knots,  c[4][N] coefficients  (row = order, col = segment).
    //
    // Single definition shared by the thermocouple motion data and the
    // temperature-dependent material properties.
    // -----------------------------------------------------------------------
    template <int N>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        eval_piecewise_cubic(amrex::Real x,
                             double const (&xs)[N + 1],
                             double const (&c)[4][N]) noexcept
    {
        int idx = 0;
        for (int i = 0; i < N; ++i)
        {
            if (x >= xs[i] && x < xs[i + 1])
            {
                idx = i;
                break;
            }
            if (x >= xs[N])
            {
                idx = N - 1;
            }
        }
        double const dx = static_cast<double>(x) - xs[idx];
        return static_cast<amrex::Real>(((c[0][idx] * dx + c[1][idx]) * dx + c[2][idx]) * dx + c[3][idx]);
    }

} // namespace cryo_interp

#endif
