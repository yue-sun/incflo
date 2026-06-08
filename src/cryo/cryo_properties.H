#ifndef INCFLO_CRYO_PROPERTIES_H_
#define INCFLO_CRYO_PROPERTIES_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

#include <cryo_interp.H>

// Material properties for the cryo simulation: bulk SI constants, the
// physical -> simulation unit conversions, and the temperature-dependent
// (spline) thermal properties. This header is free of any incflo coupling;
// incflo members in cryo.H are wired up from the constants and helpers here.

namespace cryo_props
{

    ///////////////////////////////////////////////////////////////////////////
    //
    // Simulation scales (SI units)
    //
    ///////////////////////////////////////////////////////////////////////////

    constexpr double l_phys = 1.e-3;    // length scale [m]  (mm)
    constexpr double t_phys = 1.e-3;    // time scale [s]    (ms)
    constexpr double m_phys = 6.516e-7; // mass scale [kg]   (CHR units)

    ///////////////////////////////////////////////////////////////////////////
    //
    // Unit conversion: physical (SI) -> simulation units
    //
    //   l, t, m are the simulation length [m], time [s] and mass [kg] scales.
    //   Multiply an SI quantity by the matching factor to get sim units.
    //
    ///////////////////////////////////////////////////////////////////////////

    constexpr double conv_rho_factor(double l, double /*t*/, double m) noexcept
    {
        return l * l * l / m; // density
    }
    constexpr double conv_cp_factor(double l, double t, double /*m*/) noexcept
    {
        return t * t / (l * l); // specific heat
    }
    constexpr double conv_vhc_factor(double l, double t, double m) noexcept
    {
        return l * t * t / m; // volumetric heat capacity (rho*cp)
    }
    constexpr double conv_k_factor(double l, double t, double m) noexcept
    {
        return t * t * t / (m * l); // thermal conductivity
    }

    // Conversion factors evaluated at the simulation scales above
    constexpr double conv_rho = conv_rho_factor(l_phys, t_phys, m_phys);
    constexpr double conv_cp = conv_cp_factor(l_phys, t_phys, m_phys);
    constexpr double conv_vhc = conv_vhc_factor(l_phys, t_phys, m_phys);
    constexpr double conv_k = conv_k_factor(l_phys, t_phys, m_phys);

    ///////////////////////////////////////////////////////////////////////////
    //
    // Bulk material constants in SI units
    //   rho [kg/m^3]   cp [J/(kg K)]   k [W/(m K)]
    //
    ///////////////////////////////////////////////////////////////////////////

    // Liquid ethane
    constexpr double rho_eth_phys = 651.6;
    constexpr double cp_eth_phys = 2325.8;  // NIST liquid ethane at 90 K
    constexpr double k_eth_phys = 0.25548;  // NIST liquid ethane at 90 K
    // Thermocouple
    constexpr double rho_tcp_phys = 8860.0;
    constexpr double cp_tcp_phys = 393.0;
    constexpr double k_tcp_phys = 21.8;
    // Plunger (EM grid)
    constexpr double rho_plu_phys = 19300.0;
    constexpr double cp_plu_phys = 125.0;
    constexpr double k_plu_phys = 345.0;
    // Sample (water)
    // TODO(sample-T-props): these are constant water values at a single
    // temperature; replace with temperature-dependent cp_water(T)/lam_water(T)
    // splines (see thermal_properties below + the ethane interpolants) and wire
    // them into incflo_specific_heat.cpp / incflo_rheology.cpp.
    constexpr double rho_sam_phys = 1000.0;
    constexpr double cp_sam_phys = 4187.0;
    constexpr double k_sam_phys = 0.598;
    // Sapphire disk
    constexpr double rho_sap_phys = 3980.0;
    constexpr double cp_sap_phys = 700.0; // TODO: fix average
    constexpr double k_sap_phys = 40.0;
    // Diamond disk
    constexpr double rho_dia_phys = 3515.0;
    constexpr double cp_dia_phys = 500.0;
    constexpr double k_dia_phys = 1800.0; // TODO: fix average

    ///////////////////////////////////////////////////////////////////////////
    //
    // Convert bulk material constants in simulation units
    //
    ///////////////////////////////////////////////////////////////////////////
    // TODO: check if this is necessary given we're doing spline interpolation for the thermal properties; if so, add the conversion to the spline coefficients instead of doing it here.

    // Liquid ethane
    constexpr amrex::Real rho_eth = rho_eth_phys * conv_rho;
    constexpr amrex::Real cp_eth = cp_eth_phys * conv_cp;
    constexpr amrex::Real vhc_eth = rho_eth_phys * cp_eth_phys * conv_vhc;
    constexpr amrex::Real k_eth = k_eth_phys * conv_k;
    constexpr amrex::Real kappa_eth = k_eth / vhc_eth;
    // Thermocouple
    constexpr amrex::Real rho_tcp = rho_tcp_phys * conv_rho;
    constexpr amrex::Real cp_tcp = cp_tcp_phys * conv_cp;
    constexpr amrex::Real vhc_tcp = rho_tcp_phys * cp_tcp_phys * conv_vhc;
    constexpr amrex::Real k_tcp = k_tcp_phys * conv_k;
    constexpr amrex::Real kappa_tcp = k_tcp / vhc_tcp;
    // Plunger (EM grid)
    constexpr amrex::Real rho_plu = rho_plu_phys * conv_rho;
    constexpr amrex::Real cp_plu = cp_plu_phys * conv_cp;
    constexpr amrex::Real vhc_plu = rho_plu_phys * cp_plu_phys * conv_vhc;
    constexpr amrex::Real k_plu = k_plu_phys * conv_k;
    constexpr amrex::Real kappa_plu = k_plu / vhc_plu;
    // Sample
    constexpr amrex::Real rho_sam = rho_sam_phys * conv_rho;
    constexpr amrex::Real cp_sam = cp_sam_phys * conv_cp;
    constexpr amrex::Real vhc_sam = rho_sam_phys * cp_sam_phys * conv_vhc;
    constexpr amrex::Real k_sam = k_sam_phys * conv_k;
    constexpr amrex::Real kappa_sam = k_sam / vhc_sam;
    // Sapphire disk
    constexpr amrex::Real rho_sap = rho_sap_phys * conv_rho;
    constexpr amrex::Real cp_sap = cp_sap_phys * conv_cp;
    constexpr amrex::Real vhc_sap = rho_sap_phys * cp_sap_phys * conv_vhc;
    constexpr amrex::Real k_sap = k_sap_phys * conv_k;
    constexpr amrex::Real kappa_sap = k_sap / vhc_sap;
    // Diamond disk
    constexpr amrex::Real rho_dia = rho_dia_phys * conv_rho;
    constexpr amrex::Real cp_dia = cp_dia_phys * conv_cp;
    constexpr amrex::Real vhc_dia = rho_dia_phys * cp_dia_phys * conv_vhc;
    constexpr amrex::Real k_dia = k_dia_phys * conv_k;
    constexpr amrex::Real kappa_dia = k_dia / vhc_dia;

    ///////////////////////////////////////////////////////////////////////////
    //
    // Temperature-dependent thermal properties (cubic splines)
    //
    ///////////////////////////////////////////////////////////////////////////

    // -----------------------------------------------------------------------
    // Type-K TC (Chromel + Alumel average), 90-290 K
    //   cp_typeK  : J / (g·K)
    //   lam_typeK : W / (m·K)
    // Spline knots and coefficients fitted from Chung & Brill (1993) and
    // Sundqvist (1992); see theory/tc_properties.ipynb for derivation.
    // -----------------------------------------------------------------------
    namespace thermal_properties
    {
        static constexpr double cp_knots[10] = {
            90., 100., 125., 150., 175., 200., 225., 250., 275., 290.};
        static constexpr double cp_c[4][9] = {
            {-2.90577482e-07, -2.90577482e-07, +2.01293765e-07, -6.78285054e-08, +3.50179982e-08, +3.49118649e-08, +5.49395759e-08, -9.48646471e-08, -9.48646471e-08},
            {+1.23071413e-05, +3.58981688e-06, -1.82034943e-05, -3.10646189e-06, -8.19359979e-06, -5.56724992e-06, -2.94886005e-06, +1.17160814e-06, -5.94324039e-06},
            {+2.19490020e-03, +2.35386978e-03, +1.98852784e-03, +1.45577894e-03, +1.17327740e-03, +8.29256156e-04, +6.16353406e-04, +5.71922109e-04, +4.52631302e-04},
            {+2.07254507e-01, +2.30143645e-01, +2.86693752e-01, +3.28174979e-01, +3.61568094e-01, +3.86326185e-01, +4.04123556e-01, +4.18547784e-01, +4.32095832e-01},
        };

        static constexpr double lam_knots[10] = {
            90., 100., 125., 150., 175., 200., 225., 250., 275., 290.};
        static constexpr double lam_c[4][9] = {
            {+1.22113173e-06, +1.22113173e-06, +2.38465207e-06, -4.35974002e-06, -9.45691994e-07, +1.74250800e-06, +3.75660008e-07, -4.51480277e-08, -4.51480277e-08},
            {-1.02762836e-04, -6.61288843e-05, +2.54559958e-05, +2.04304901e-04, -1.22675600e-04, -1.93602500e-04, -6.29144001e-05, -3.47398996e-05, -3.81260016e-05},
            {+4.05789320e-02, +3.88900148e-02, +3.78731926e-02, +4.36172150e-02, +4.56579475e-02, +3.77509950e-02, +3.13380725e-02, +2.88967150e-02, +2.70750675e-02},
            {+1.57532658e+01, +1.61500000e+01, +1.71000000e+01, +1.81000000e+01, +1.92500000e+01, +2.03000000e+01, +2.11500000e+01, +2.19000000e+01, +2.26000000e+01},
        };

        // -------------------------------------------------------------------
        // Liquid ethane, NIST WebBook (C74840), 1 atm isobar, 90.4-184.6 K
        //   cp_eth  : J / (g K)      lam_eth : W / (m K)
        // Cubic spline (not-a-knot); see theory/material_properties.ipynb.
        // -------------------------------------------------------------------
        static constexpr double cp_eth_knots[11] = {
            90.368, 100.370, 110.370, 120.370, 130.370, 140.370, 150.370, 160.370, 170.370, 180.370, 184.570};
        static constexpr double cp_eth_c[4][10] = {
            {-3.70949769e-06, -3.70949769e-06, -1.03871073e-06, -5.35659386e-07, -1.86517272e-08, +1.02662944e-08, +1.77586550e-07, +1.79387507e-07, +2.04863421e-07, +2.04863421e-07},
            {+2.87238184e-04, +1.75930996e-04, +6.46460652e-05, +3.34847433e-05, +1.74149617e-05, +1.68554099e-05, +1.71633987e-05, +2.24909952e-05, +2.78726204e-05, +3.40185231e-05},
            {-6.90097833e-03, -2.26836019e-03, +1.37410421e-04, +1.11871851e-03, +1.62771556e-03, +1.97041927e-03, +2.31060736e-03, +2.70715130e-03, +3.21078745e-03, +3.82969889e-03},
            {+2.32580000e+00, +2.28180000e+00, +2.27300000e+00, +2.27980000e+00, +2.29380000e+00, +2.31180000e+00, +2.33320000e+00, +2.35820000e+00, +2.38770000e+00, +2.42280000e+00},
        };

        static constexpr double lam_eth_knots[11] = {
            90.368, 100.370, 110.370, 120.370, 130.370, 140.370, 150.370, 160.370, 170.370, 180.370, 184.570};
        static constexpr double lam_eth_c[4][10] = {
            {+2.19018048e-08, +2.19018048e-08, +2.20327376e-08, +2.99672448e-08, +1.80982832e-08, +7.63962246e-09, +2.13432270e-08, -3.01253041e-09, +1.07068946e-08, +1.07068946e-08},
            {-3.81489436e-06, -3.15770881e-06, -2.50065466e-06, -1.83967254e-06, -9.40655192e-07, -3.97706696e-07, -1.68518023e-07, +4.71778787e-07, +3.81402875e-07, +7.02609714e-07},
            {-7.70873116e-04, -8.40613092e-04, -8.97196727e-04, -9.40599999e-04, -9.68403276e-04, -9.81786895e-04, -9.87449142e-04, -9.84416535e-04, -9.75884718e-04, -9.65044592e-04},
            {+2.55480000e-01, +2.47410000e-01, +2.38710000e-01, +2.29510000e-01, +2.19950000e-01, +2.10190000e-01, +2.00340000e-01, +1.90470000e-01, +1.80670000e-01, +1.70960000e-01},
        };
    } // namespace thermal_properties

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        cp_typeK(amrex::Real T_K) noexcept
    {
        return cryo_interp::eval_piecewise_cubic(T_K, thermal_properties::cp_knots, thermal_properties::cp_c);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        lam_typeK(amrex::Real T_K) noexcept
    {
        return cryo_interp::eval_piecewise_cubic(T_K, thermal_properties::lam_knots, thermal_properties::lam_c);
    }

    // Liquid-ethane interpolant. The argument is clamped at T_sat = 184.57 K,
    // so above saturation the property is held flat at its last liquid value
    // (simplest "hold at last point" surrogate; swap for a vapor floor / smooth
    // clamp here if desired — see theory/k_eff_boiling.ipynb).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        cp_ethane(amrex::Real T_K) noexcept
    {
        T_K = (T_K > amrex::Real(184.57)) ? amrex::Real(184.57) : T_K;
        return cryo_interp::eval_piecewise_cubic(T_K, thermal_properties::cp_eth_knots, thermal_properties::cp_eth_c);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        lam_ethane(amrex::Real T_K) noexcept
    {
        T_K = (T_K > amrex::Real(184.57)) ? amrex::Real(184.57) : T_K;
        return cryo_interp::eval_piecewise_cubic(T_K, thermal_properties::lam_eth_knots, thermal_properties::lam_eth_c);
    }

} // namespace cryo_props

#endif
