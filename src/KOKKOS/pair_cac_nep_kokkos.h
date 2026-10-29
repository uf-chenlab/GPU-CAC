/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
#if __has_include("nep.h")
PairStyle(cac/nep/kk,PairCACNEPKokkos<LMPDeviceType>);
PairStyle(cac/nep/kk/device,PairCACNEPKokkos<LMPDeviceType>);
PairStyle(cac/nep/kk/host,PairCACNEPKokkos<LMPHostType>);
#endif
// clang-format on
#else

#ifndef LMP_PAIR_CAC_NEP_KOKKOS_H
#define LMP_PAIR_CAC_NEP_KOKKOS_H

#include "nep_kokkos_engine.h"
#include "pair_cac_nep.h"

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   CAC + NEP with the Kokkos NEP kernels.  The CAC machinery (atom style,
   comm, quadrature neighbor lists, integration) stays on the host; per step
   the nodal positions are uploaded, all lattice sites are interpolated on
   the device, the NEP kernels run on the site set, and the force densities
   and energies of the quadrature points are copied back for the nodal
   projection.  Runs without "-k on": Kokkos is initialized on first use.
------------------------------------------------------------------------- */

template <class DeviceType> class PairCACNEPKokkos : public PairCACNEP {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairCACNEPKokkos(class LAMMPS *);
  ~PairCACNEPKokkos() override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;

 protected:
  void compute_sites(int, int) override;
  void upload_site_table();
  void build_device_lists();
  int build_lists_on_host() override { return dev_lists ? 0 : 1; }

  NEPKokkosEngine<DeviceType> engine;

  // site table on the device (rebuilt with the neighbor lists)
  bigint dev_table_ncalls;
  int nnode_rows;                                           // rows of the flattened node array
  std::vector<int> pair_base_host;                          // node row of (owner,poly) per site
  std::vector<std::pair<int, int>> node_rows;               // (owner,poly) of every 8-row block or atom row
  Kokkos::View<int *, DeviceType> d_site_base;              // first node row of the site's (owner,poly)
  Kokkos::View<int *, DeviceType> d_site_isatom;
  Kokkos::View<double *[8], Kokkos::LayoutRight, DeviceType> d_site_w;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> d_nodes;
  typename Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>::host_mirror_type h_nodes;
  typename AT::t_kkfloat_1d_3_lr d_x;
  typename AT::t_int_1d d_type;
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_numneigh;
  typename AT::t_int_1d d_ilist;
  typename AT::t_kkacc_1d_3 d_f;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;
  Kokkos::View<int *, DeviceType> d_quad_site;
  Kokkos::View<double *[4], Kokkos::LayoutRight, DeviceType> d_quad_out;
  typename Kokkos::View<double *[4], Kokkos::LayoutRight, DeviceType>::host_mirror_type h_quad_out;
  int max_site_neigh, neigh_cols;
  int dev_lists;                  // 1: centers and neighbor lists are built on the device
  int ncenters;                   // number of centers (device list build)
  bool lists_pending;             // rebuild: lists must be (re)built after the next position update
  Kokkos::View<double *, DeviceType> d_xs, d_xb;            // site coordinates (site order / cell order)
  Kokkos::View<int *, DeviceType> d_order, d_bstart, d_sbin, d_isquad, d_has;
  bool static_valid;                                        // cache_* describe the uploaded static site arrays
  std::vector<int> cache_owner, cache_poly, cache_type, cache_quad_site;
  std::vector<double> cache_w;
  Kokkos::View<int *, DeviceType> d_slist, d_sfirst;        // flat CSR site lists on the device
};

}    // namespace LAMMPS_NS

#endif
#endif
