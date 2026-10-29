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
PairStyle(cac/lj/kk,PairCACLJKokkos<LMPDeviceType>);
PairStyle(cac/lj/kk/device,PairCACLJKokkos<LMPDeviceType>);
PairStyle(cac/lj/kk/host,PairCACLJKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_CAC_LJ_KOKKOS_H
#define LMP_PAIR_CAC_LJ_KOKKOS_H

#include "cac_site_device.h"
#include "pair_cac_lj.h"

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   CAC + Lennard-Jones, force densities of the quadrature points on the device.
   Run without "-k on": Kokkos is initialized on first use (see CACSiteDevice).
------------------------------------------------------------------------- */

template <class DeviceType> class PairCACLJKokkos : public PairCACLJ {
 public:
  PairCACLJKokkos(class LAMMPS *);
  ~PairCACLJKokkos() override;
  void init_style() override;

 protected:
  void compute_sites(int, int) override;
  int build_lists_on_host() override { return 0; }

  CACSiteDevice<DeviceType> dev;
  bool tables_ready;
  Kokkos::View<double *, DeviceType> d_lj1, d_lj2, d_lj3, d_lj4, d_offset;    // [(itype)*(ntypes+1) + jtype]
};

}    // namespace LAMMPS_NS

#endif
#endif
