target_include_directories(incflo PRIVATE ${CMAKE_CURRENT_LIST_DIR})

if (SIM_CRYO)
   target_sources(incflo
      PRIVATE
      cryo.cpp
      cryo.H
      cryo_sample.cpp
      cryo_sample.H
      cryo_tc_experiment.cpp
      cryo_tc_experiment.H
      cryo_properties.H
      cryo_interp.H)
endif()
