#ifndef INCFLO_CRYO_TC_MOTION_H_
#define INCFLO_CRYO_TC_MOTION_H_

#include <AMReX_REAL.H>

namespace cryo_tc
{

    // Prescribed thermocouple-bead kinematics measured in the linear motor
    // experiment. The spline data lives in cryo_tc_experiment.cpp; material
    // thermal properties are in cryo_properties.H, not here.
    struct ThermocoupleMotion
    {
        amrex::Real speed = amrex::Real(0.0);
        amrex::Real depth = amrex::Real(0.0);
    };

    ThermocoupleMotion evaluate_motion(amrex::Real time);

} // namespace cryo_tc

#endif
