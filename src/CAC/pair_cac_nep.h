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

// pair_style cac/nep requires the NEP package as well as CAC

#ifdef PAIR_CLASS
// clang-format off
#if __has_include("nep.h")
PairStyle(cac/nep,PairCACNEP);
#endif
// clang-format on
#else

#ifndef LMP_PAIR_CAC_NEP_H
#define LMP_PAIR_CAC_NEP_H

#include "nep.h"
#include "pair_cac.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairCACNEP : public PairCAC {
 public:
  PairCACNEP(class LAMMPS *);
  ~PairCACNEP() override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  NEP3 nep_model;
  int *type_map;      // LAMMPS atom type -> NEP species index
  double cutmax;      // largest NEP cutoff (radial or angular)

 protected:
  void allocate() override;
  void compute_sites(int, int) override;
  void force_densities(int, double, double, double, double, double &, double &,
                       double &) override;

  // per-step NEP input/output in site space
  std::vector<int> centers;         // sites whose descriptors are evaluated
  std::vector<int> site_numneigh;   // site neighbor counts (0 for non-centers)
  std::vector<int *> site_firstneigh;
  std::vector<double *> site_xptr, site_fptr;
  std::vector<double> site_f, site_e, site_v;
  std::vector<double *> site_vptr;
  bigint centers_ncalls;
  int legacy_lists;    // CAC_LEGACY_LISTS set: old 2rc outer lists instead of site enumeration
  void build_centers();
};

}    // namespace LAMMPS_NS

#endif
#endif
