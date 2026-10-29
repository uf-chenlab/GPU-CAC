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
PairStyle(cac/eam/kk,PairCACEAMKokkos<LMPDeviceType>);
PairStyle(cac/eam/kk/device,PairCACEAMKokkos<LMPDeviceType>);
PairStyle(cac/eam/kk/host,PairCACEAMKokkos<LMPHostType>);
PairStyle(cac/eam/alloy/kk,PairCACEAMAlloyKokkos<LMPDeviceType>);
PairStyle(cac/eam/alloy/kk/device,PairCACEAMAlloyKokkos<LMPDeviceType>);
PairStyle(cac/eam/alloy/kk/host,PairCACEAMAlloyKokkos<LMPHostType>);
PairStyle(cac/eam/fs/kk,PairCACEAMFSKokkos<LMPDeviceType>);
PairStyle(cac/eam/fs/kk/device,PairCACEAMFSKokkos<LMPDeviceType>);
PairStyle(cac/eam/fs/kk/host,PairCACEAMFSKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_CAC_EAM_KOKKOS_H
#define LMP_PAIR_CAC_EAM_KOKKOS_H

#include "cac_site_device.h"
#include "pair_cac_eam.h"
#include "pair_cac_eam_alloy.h"
#include "pair_cac_eam_fs.h"

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   CAC + EAM on the device: the embedding density and its derivative of every center, then
   the force density, energy and virial of every quadrature point.  The three styles only
   differ in how the potential file is read (host); the device code is shared.
------------------------------------------------------------------------- */

template <class DeviceType> class PairCACEAMKokkosCore {
 public:
  void init(PairCACEAM &pair, LAMMPS *lmp);
  void compute(PairCACEAM &pair);

 private:
  CACSiteDevice<DeviceType> dev;
  bool tables_ready = false;
  Kokkos::View<double *[7], Kokkos::LayoutRight, DeviceType> d_frho, d_rhor, d_z2r;    // [type index * (n+1) + m]
  Kokkos::View<int *, DeviceType> d_type2frho, d_type2rhor, d_type2z2r;
  Kokkos::View<double *, DeviceType> d_scale, d_rho, d_fp;
};

template <class DeviceType> class PairCACEAMKokkos : public PairCACEAM {
 public:
  PairCACEAMKokkos(class LAMMPS *lmp) : PairCACEAM(lmp) {}
  void init_style() override
  {
    PairCACEAM::init_style();
    core.init(*this, lmp);
  }

 protected:
  void compute_sites(int, int) override { core.compute(*this); }
  int build_lists_on_host() override { return 0; }
  PairCACEAMKokkosCore<DeviceType> core;
};

template <class DeviceType> class PairCACEAMAlloyKokkos : public PairCACEAMAlloy {
 public:
  PairCACEAMAlloyKokkos(class LAMMPS *lmp) : PairCACEAM(lmp), PairCACEAMAlloy(lmp) {}
  void init_style() override
  {
    PairCACEAM::init_style();
    core.init(*this, lmp);
  }

 protected:
  void compute_sites(int, int) override { core.compute(*this); }
  int build_lists_on_host() override { return 0; }
  PairCACEAMKokkosCore<DeviceType> core;
};

template <class DeviceType> class PairCACEAMFSKokkos : public PairCACEAMFS {
 public:
  PairCACEAMFSKokkos(class LAMMPS *lmp) : PairCACEAM(lmp), PairCACEAMFS(lmp) {}
  void init_style() override
  {
    PairCACEAM::init_style();
    core.init(*this, lmp);
  }

 protected:
  void compute_sites(int, int) override { core.compute(*this); }
  int build_lists_on_host() override { return 0; }
  PairCACEAMKokkosCore<DeviceType> core;
};

}    // namespace LAMMPS_NS

#endif
#endif
