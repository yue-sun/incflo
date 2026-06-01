
#ifndef CRYO_
#define CRYO_

// This header is included in incflo.H.
// These functions are members of incflo class.

///////////////////////////////////////////////////////////////////////////
//
// Set geometry
//
///////////////////////////////////////////////////////////////////////////

// Debug sphere
amrex::Real rdebug = amrex::Real(0.5); // debug sphere radius: 0.5mm

// EM grid (gold)
amrex::Real Remg = amrex::Real(1.5);         // EM grid outer radius: 1.5mm
amrex::Real wemg = amrex::Real(0.025 / 2);   // EM grid width: 25um
amrex::Real remg = Remg - amrex::Real(0.24); // EM grid ring width: 240um

// Sapphire disk
amrex::Real Rsap = amrex::Real(1.5);      // sapphire disk radius: 1.5mm
amrex::Real wsap = amrex::Real(0.16 / 2); // sapphire disk width: 160um

// Sample
amrex::Real wsam = amrex::Real(0.01); // sample width: 10um

///////////////////////////////////////////////////////////////////////////
//
// Set temperature and thermal properties
//
///////////////////////////////////////////////////////////////////////////

// Temperatures K
amrex::Real temp_eth = amrex::Real(90.0);
amrex::Real temp_tcp = amrex::Real(244.0); // 288
amrex::Real temp_plu = amrex::Real(244.0);
amrex::Real temp_sam = amrex::Real(244.0);

// Set physical scales in SI units
amrex::Real l_phys = amrex::Real(1.e-3); // mm
amrex::Real t_phys = amrex::Real(1.e-3); // ms
// amrex::Real m_phys = amrex::Real(651.6e-9); // kg/mm^3
amrex::Real m_phys = amrex::Real(6.516e-7); // CHR units

// Cryo-plunging properties
// Density kg/m^3
amrex::Real rho_eth_phys = amrex::Real(651.6);
amrex::Real rho_tcp_phys = amrex::Real(8860.0);
amrex::Real rho_plu_phys = amrex::Real(19300.0);
amrex::Real rho_sam_phys = amrex::Real(1000.0);
amrex::Real rho_sap_phys = amrex::Real(3980.0);
amrex::Real rho_dia_phys = amrex::Real(3515.0);
// Specific heat capacity J/(kg K)
amrex::Real cp_eth_phys = amrex::Real(1610.0);
amrex::Real cp_tcp_phys = amrex::Real(393.0);
amrex::Real cp_plu_phys = amrex::Real(125.0);
amrex::Real cp_sam_phys = amrex::Real(4187.0);
amrex::Real cp_sap_phys = amrex::Real(700.0); //TODO: fix average
amrex::Real cp_dia_phys = amrex::Real(500.0);
// Thermal conductivity W/(m K)
amrex::Real k_eth_phys = amrex::Real(0.256);
amrex::Real k_tcp_phys = amrex::Real(21.8);
amrex::Real k_plu_phys = amrex::Real(345.0);
amrex::Real k_sam_phys = amrex::Real(0.598);
amrex::Real k_sap_phys = amrex::Real(40.0);
amrex::Real k_dia_phys = amrex::Real(1800.0); // TODO: fix average

///////////////////////////////////////////////////////////////////////////
//
// Helper functions
//
///////////////////////////////////////////////////////////////////////////

void cryo_init();

void cryo_update(amrex::Real time = amrex::Real(-1.0));

void cryo_set_geom_velocity(int i, int j, int k,
                            amrex::Real x, amrex::Real y, amrex::Real z,
                            amrex::Real &velx, amrex::Real &vely, amrex::Real &velz,
                            int &cell_type_ijk, amrex::Real &rho_ijk,
                            amrex::Real time,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi);

void cryo_set_temp_top_bc(amrex::Real z, amrex::Real &temperature_ijk,
                          int cell_type_ijk,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi);

void cryo_set_thermal(int i, int j, int k,
                      amrex::Real x, amrex::Real y, amrex::Real z,
                      int cell_type_ijk, amrex::Real time,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &problo,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &probhi);

void cryo_load_emgrid();

void cryo_load_disk();

void cryo_load_sample();

///////////////////////////////////////////////////////////////////////////
//
// Convert physical units to simulation units
//
///////////////////////////////////////////////////////////////////////////

// Calculate conversion factors
amrex::Real l2_phys = l_phys * l_phys;
amrex::Real l3_phys = l2_phys * l_phys;
amrex::Real t2_phys = t_phys * t_phys;
amrex::Real conv_rho = l3_phys / m_phys;
amrex::Real conv_cp = t2_phys / l2_phys;
amrex::Real conv_vhc = l_phys * t2_phys / m_phys;
amrex::Real conv_k = t2_phys * t_phys / (m_phys * l_phys);

// Convert physical constants into simulation units
// Liquid ethane
amrex::Real rho_eth = rho_eth_phys * conv_rho;
amrex::Real cp_eth = cp_eth_phys * conv_cp;
amrex::Real vhc_eth = rho_eth_phys * cp_eth_phys * conv_vhc;
amrex::Real k_eth = k_eth_phys * conv_k;
amrex::Real kappa_eth = k_eth / vhc_eth;
// Thermocouple
amrex::Real rho_tcp = rho_tcp_phys * conv_rho;
amrex::Real cp_tcp = cp_tcp_phys * conv_cp;
amrex::Real vhc_tcp = rho_tcp_phys * cp_tcp_phys * conv_vhc;
amrex::Real k_tcp = k_tcp_phys * conv_k;
amrex::Real kappa_tcp = k_tcp / vhc_tcp;
// Plunger (EM grid)
amrex::Real rho_plu = rho_plu_phys * conv_rho;
amrex::Real cp_plu = cp_plu_phys * conv_cp;
amrex::Real vhc_plu = rho_plu_phys * cp_plu_phys * conv_vhc;
amrex::Real k_plu = k_plu_phys * conv_k;
amrex::Real kappa_plu = k_plu / vhc_plu;
// Sample
amrex::Real rho_sam = rho_sam_phys * conv_rho;
amrex::Real cp_sam = cp_sam_phys * conv_cp;
amrex::Real vhc_sam = rho_sam_phys * cp_sam_phys * conv_vhc;
amrex::Real k_sam = k_sam_phys * conv_k;
amrex::Real kappa_sam = k_sam / vhc_sam;
// Sapphire disk
amrex::Real rho_sap = rho_sap_phys * conv_rho;
amrex::Real cp_sap = cp_sap_phys * conv_cp;
amrex::Real vhc_sap = rho_sap_phys * cp_sap_phys * conv_vhc;
amrex::Real k_sap = k_sap_phys * conv_k;
amrex::Real kappa_sap = k_sap / vhc_sap;
// Diamond disk
amrex::Real rho_dia = rho_dia_phys * conv_rho;
amrex::Real cp_dia = cp_dia_phys * conv_cp;
amrex::Real vhc_dia = rho_dia_phys * cp_dia_phys * conv_vhc;
amrex::Real k_dia = k_dia_phys * conv_k;
amrex::Real kappa_dia = k_dia / vhc_dia;

#endif
