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
PairStyle(cac/sw/kk,PairCACSWKokkos<LMPDeviceType>);
PairStyle(cac/sw/kk/device,PairCACSWKokkos<LMPDeviceType>);
PairStyle(cac/sw/kk/host,PairCACSWKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_CAC_SW_KOKKOS_H
#define LMP_PAIR_CAC_SW_KOKKOS_H

#include "cac_site_device.h"
#include "pair_cac_sw.h"

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   CAC + Stillinger-Weber on the device: every center evaluates its pair and three-body terms once
   and scatters (atomically) the forces, energies and virials onto the participating sites; the
   quadrature points then gather their site values.  Same decomposition as PairCACSW::compute_sites.
------------------------------------------------------------------------- */

template <class DeviceType> class PairCACSWKokkos : public PairCACSW {
 public:
  PairCACSWKokkos(class LAMMPS *);
  ~PairCACSWKokkos() override;
  void init_style() override;

 protected:
  void compute_sites(int, int) override;
  int build_lists_on_host() override { return 0; }

  CACSiteDevice<DeviceType> dev;
  bool tables_ready;
  Kokkos::View<PairCACSW::Param *, DeviceType> d_params;
  Kokkos::View<int *, DeviceType> d_e2p, d_map, d_err;
};

}    // namespace LAMMPS_NS

#endif
#endif
