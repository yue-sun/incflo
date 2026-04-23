target_include_directories(incflo PRIVATE ${CMAKE_CURRENT_LIST_DIR})

if (SIM_CRYO)
   target_sources(incflo
      PRIVATE
      cryo.cpp
      cryo.H)
endif()
