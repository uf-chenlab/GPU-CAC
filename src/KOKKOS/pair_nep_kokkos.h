/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Copyright (2003) Sandia Corporation.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nep/kk, PairNEPKokkos<LMPDeviceType>)
    PairStyle(nep/kk/device, PairNEPKokkos<LMPDeviceType>)
        PairStyle(nep/kk/host, PairNEPKokkos<LMPHostType>)
#else

#ifndef LMP_PAIR_NEP_KOKKOS_H
#define LMP_PAIR_NEP_KOKKOS_H

#include "neigh_list_kokkos.h"
#include "nep_kokkos_engine.h"
#include "pair_nep.h"
#include "pair_kokkos.h"
#include <algorithm>

namespace LAMMPS_NS {

template <class DeviceType> class PairNEPKokkos : public PairNEP {
 public:
  enum { EnabledNeighFlags = FULL | HALF };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairNEPKokkos(class LAMMPS *);
  PairNEPKokkos(const PairNEPKokkos &pair);
  virtual ~PairNEPKokkos();

  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void init_style();
  virtual void coeff(int, char **);

 protected:
  NEPKokkosEngine<DeviceType> engine;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_6 v_virial;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  DAT::ttransform_kkacc_1d_9 k_cvatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;
  typename AT::t_kkacc_1d_9 d_cvatom;

  typename AT::t_kkacc_1d d_eatom_tmp;
  typename AT::t_kkacc_1d_6 d_vatom_tmp;

  int neighflag, newton_pair;
  int nlocal, nall, eflag, vflag, inum;

  friend void pair_virial_fdotr_compute<PairNEPKokkos>(PairNEPKokkos *);
  void cleanup_copy();
};
}    // namespace LAMMPS_NS
#endif
#endif
