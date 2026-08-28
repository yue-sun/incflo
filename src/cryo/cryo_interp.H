#ifndef INCFLO_CRYO_INTERP_H_
#define INCFLO_CRYO_INTERP_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>

namespace cryo_interp
{

    // -----------------------------------------------------------------------
    // Piecewise cubic evaluator
    //     f(x) = c[0]·dx³ + c[1]·dx² + c[2]·dx + c[3]
    // where dx = x - left knot of the containing segment.
    //
    // Layout: xs[N+1] knots, c[4][N] coefficients (col = segment; row 0 is the
    // CUBIC coefficient and row 3 the constant term, i.e. scipy
    // CubicSpline/PPoly `.c` ordering, highest order first). The generators in
    // theory/*.ipynb emit the rows in that order -- reversing them produces a
    // smooth, plausible, and completely wrong property with no crash to reveal
    // it.
    //
    // The knot/coefficient tables are function-local `constexpr` arrays in each
    // accessor (cryo_properties.H, cryo_grid.H) rather than namespace-scope
    // variables, so that passing them here by reference stays legal in device
    // code: CUDA forbids taking the address of a host constexpr non-scalar in
    // a __device__ function.
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
        // Clamp the argument to the spline's knot range so out-of-range
        // inputs return the boundary value (flat extrapolation) instead of an
        // unbounded cubic extrapolation. Without this, a temperature that
        // overshoots outside [xs[0], xs[N]] -- e.g. a sharp advective
        // under/overshoot at a material interface -- produces a garbage or
        // negative cp(T)/k(T), which makes the implicit temperature-diffusion
        // operator non-SPD and blows up the MLMG solve. (Callers that already
        // clamp, such as the thermocouple-motion spline with xs[0]=0, are
        // unaffected since this clamp is to the same range.)
        {
            amrex::Real const xlo = static_cast<amrex::Real>(xs[0]);
            amrex::Real const xhi = static_cast<amrex::Real>(xs[N]);
            x = (x < xlo) ? xlo : ((x > xhi) ? xhi : x);
        }

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

    // -----------------------------------------------------------------------
    // Closed-form (non-spline) material laws used by cryo_grid.H.
    // -----------------------------------------------------------------------

    // Sapphire (alpha-Al2O3) MOLAR heat capacity from the Ditmars et al. (1982,
    // J. Res. NBS 87, 159) / Chang least-squares spline fit:
    //   Cp(T) = sum_{i=0..6} A_i/i! * (T - T0)^i   over three intervals,
    // plus the derivative of the high-temperature enthalpy function (eq. 2)
    // above 273.15 K.  Returns Cp in J/(mol K).  Valid ~8.6-2250 K.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        double cp_sapphire_func(double T) noexcept
    {
        constexpr double inv_fact[7] =
            {1.0, 1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0};
        if (T < 45.0)            // 8.61 - 45 K : Cp = exp(P1)
        {
            constexpr double A[7] = {-0.5147e+01, 0.34127e+00, -0.333446e-01,
                                     0.450764e-02, -0.51464e-03, 0.397864e-04, -0.152136e-05};
            double const dt = T - 8.61;
            double p = 1.0, s = 0.0;
            for (int i = 0; i < 7; ++i) { s += A[i] * inv_fact[i] * p; p *= dt; }
            return std::exp(s);
        }
        else if (T < 125.0)      // 45 - 125 K : P2
        {
            constexpr double A[7] = {0.6966e+00, 0.59387e-01, 0.40357e-02,
                                     0.95173e-04, -0.35910e-05, -0.6498e-07, 0.4089e-08};
            double const dt = T - 40.0;
            double p = 1.0, s = 0.0;
            for (int i = 0; i < 7; ++i) { s += A[i] * inv_fact[i] * p; p *= dt; }
            return s;
        }
        else if (T < 273.15)     // 125 - 273.15 K : P3
        {
            constexpr double A[7] = {0.21993e+02, 0.38853e+00, 0.13955e-02,
                                     -0.83967e-04, 0.19133e-05, -0.31778e-07, 0.29562e-09};
            double const dt = T - 125.0;
            double p = 1.0, s = 0.0;
            for (int i = 0; i < 7; ++i) { s += A[i] * inv_fact[i] * p; p *= dt; }
            return s;
        }
        else                     // > 273.15 K : dH/dT of the eq.(2) enthalpy fit
        {
            constexpr double A = 6.6253e+07, B = -4.54238e+06, C = -5.475599e+04,
                             D = 2.574076e+02, E = -8.57516e-02, F = 4.299063e-05,
                             G = -1.15192e-08, H5 = 1.26351e-12;
            return -2.0 * A / (T * T * T) - B / (T * T) + C / T + D
                   + 2.0 * E * T + 3.0 * F * T * T + 4.0 * G * T * T * T + 5.0 * H5 * T * T * T * T;
        }
    }

    // Sapphire thermal conductivity, Brown (2010) functional form, eq.(1):
    //   kappa(T) [W/cm/K] = a / (ln(b T))^c - d / T ,   then x100 -> W/(m K).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        double k_sapphire_func(double T) noexcept
    {
        constexpr double a = 1.0e6, b = 0.138897, c = 8.1382, d = -1022.11;
        double const lx = std::log(b * T);
        return (a / std::pow(lx, c) - d / T);
    }

} // namespace cryo_interp

#endif
