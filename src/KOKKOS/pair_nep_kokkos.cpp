/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Copyright (2003) Sandia Corporation.
------------------------------------------------------------------------- */

#include "pair_nep_kokkos.h"
#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "nep_kokkos.h"
#include "suffix.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

template <class DeviceType> PairNEPKokkos<DeviceType>::PairNEPKokkos(LAMMPS *lmp) : PairNEP(lmp)
{
  suffix_flag |= Suffix::KOKKOS;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  centroidstressflag = CENTROID_NOTAVAIL;

  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

template <class DeviceType>
PairNEPKokkos<DeviceType>::PairNEPKokkos(const PairNEPKokkos<DeviceType> &pair) : PairNEP(pair)
{
  atomKK = pair.atomKK;
  execution_space = pair.execution_space;
  datamask_read = pair.datamask_read;
  datamask_modify = pair.datamask_modify;

  // shallow copies of Kokkos views are intended
  engine = pair.engine;
  k_eatom = pair.k_eatom;
  k_vatom = pair.k_vatom;
  k_cvatom = pair.k_cvatom;
  copymode = 1;
  cleanup_copy();
}

template <class DeviceType> PairNEPKokkos<DeviceType>::~PairNEPKokkos()
{
  if (!copymode) {
    MemoryKokkos *memoryKK = (MemoryKokkos *) memory;
    if (eatom) memoryKK->destroy_kokkos(k_eatom, eatom);
    if (vatom) memoryKK->destroy_kokkos(k_vatom, vatom);
    if (cvatom) memoryKK->destroy_kokkos(k_cvatom, cvatom);
  }
}

/* ----------------------------------------------------------------------
   pair_style nep/kk [fp64 | fp32]
     fp64 (default): all math in double precision
     fp32 (alias mixed): descriptor/force math in single precision, forces/energies/virial
                         accumulated in double
------------------------------------------------------------------------- */
template <class DeviceType> void PairNEPKokkos<DeviceType>::settings(int narg, char **arg)
{
  engine.use_fp32 = false;
  engine.use_df64 = false;
  if (narg == 0) return;
  if (narg == 1 && (strcmp(arg[0], "fp32") == 0 || strcmp(arg[0], "mixed") == 0)) {
    engine.use_fp32 = true;
    return;
  }
  if (narg == 1 && strcmp(arg[0], "df64") == 0) {
    engine.use_df64 = true;
    return;
  }
  if (narg == 1 && (strcmp(arg[0], "fp64") == 0 || strcmp(arg[0], "double") == 0)) return;
  error->all(FLERR, "Illegal pair_style nep/kk command: expected no argument, 'fp64', 'fp32' or 'df64'");
}

template <class DeviceType> void PairNEPKokkos<DeviceType>::init_style()
{
  PairNEP::init_style();

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

template <class DeviceType> void PairNEPKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairNEP::coeff(narg, arg);    // parse the potential file with the CPU class
  engine.setup(nep_model, this->type_map, atom->ntypes, error);
}

template <class DeviceType> void PairNEPKokkos<DeviceType>::cleanup_copy()
{
  allocated = 0;
  cutsq = NULL;
  setflag = NULL;
  type_map = NULL;
  map = NULL;
  eatom = NULL;
  vatom = NULL;
  cvatom = NULL;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void PairNEPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  nep_kk_prof().start();
  eflag = eflag_in;
  vflag = vflag_in;
  if (eflag || vflag) { ev_setup(eflag, vflag); }

  atomKK->sync(execution_space, X_MASK | TYPE_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  if (atom->nghost > 0) {
    auto f_ghost = Kokkos::subview(f, std::pair<int, int>(atom->nlocal, atom->nlocal + atom->nghost),
                                   Kokkos::ALL);
    Kokkos::deep_copy(f_ghost, 0.0);
  }
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  inum = list->inum;
  const int ignum = inum + list->gnum;
  const int max_neighs = k_list->d_neighbors.extent(1);
  engine.ensure_short_lists(ignum, max_neighs);


  MemoryKokkos *memoryKK = (MemoryKokkos *) memory;

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  } else {
    if ((int) d_eatom_tmp.extent(0) < atom->nmax)
      d_eatom_tmp = typename AT::t_kkacc_1d("d_eatom_tmp", atom->nmax);
    d_eatom = d_eatom_tmp;
  }

  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  } else {
    if ((int) d_vatom_tmp.extent(0) < atom->nmax)
      d_vatom_tmp = typename AT::t_kkacc_1d_6("d_vatom_tmp", atom->nmax);
    d_vatom = d_vatom_tmp;
  }
  // per-atom virial accumulators are only touched (and need zeroing) when a virial is requested
  if (vflag_either) Kokkos::deep_copy(d_vatom, 0.0);

  nep_kk_prof().stop(6);

  engine.x = x;
  engine.f = f;
  engine.d_vatom = d_vatom;
  engine.type = type;
  engine.d_neighbors = k_list->d_neighbors;
  engine.d_numneigh = k_list->d_numneigh;
  engine.d_ilist = k_list->d_ilist;
  engine.d_eatom = d_eatom;
  engine.inum = inum;
  engine.nall = nall;
  engine.nmax = atom->nmax;
  engine.need_virial = (vflag_either != 0);
  engine.box_length = std::max(domain->xprd, std::max(domain->yprd, domain->zprd));
  engine.error = error;
  engine.run();

  nep_kk_prof().start();
  auto l_eatom = d_eatom;
  auto l_vatom = d_vatom;

  // One fused reduction for the global energy and virial (instead of 7 separate kernels + syncs)
  if (eflag_global || vflag_global) {
    EV_FLOAT ev;
    const bool do_e = eflag_global != 0, do_v = vflag_global != 0;
    Kokkos::parallel_reduce(
        "NepGlobalReduce", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int i, EV_FLOAT &u) {
          if (do_e) u.evdwl += l_eatom(i);
          if (do_v) {
            for (int k = 0; k < 6; k++) u.v[k] += l_vatom(i, k);
          }
        },
        ev);
    if (do_e) eng_vdwl += ev.evdwl;
    if (do_v)
      for (int k = 0; k < 6; k++) virial[k] += ev.v[k];
  }
  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space, F_MASK);

  if (cvflag_atom) {
    k_cvatom.template modify<DeviceType>();
    k_cvatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);
  nep_kk_prof().stop(7);
  comm->reverse_comm();
  atomKK->sync(execution_space, F_MASK);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairNEPKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairNEPKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
