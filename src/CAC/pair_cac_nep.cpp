/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   pair_style cac/nep : NEP (neuroevolution potential) for CAC models

   The force density at a quadrature point q is the force on the virtual
   atom at q,  F_q = sum_j f_qj - sum_i f_iq,  where f_ij = dU_i/dr_ij.
   The second sum needs the descriptors (dU_i/dq_i and the angular sums)
   of every neighbor i of q.  All sites that can lie within the cutoff of a
   quadrature point are "centers": their descriptors are evaluated once per
   step and their pair forces are scattered onto all their neighbors, so
   the force accumulated on a quadrature site is exactly F_q.  Sites at the
   edge of the quadrature neighbor lists may have truncated neighborhoods,
   but they only scatter onto sites within their own cutoff, which are
   never quadrature points unless the center is a complete one.
------------------------------------------------------------------------- */

#if __has_include("nep.h")

#include "pair_cac_nep.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairCACNEP::PairCACNEP(LAMMPS *lmp) : PairCAC(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  one_coeff = 1;
  legacy_lists = getenv("CAC_LEGACY_LISTS") != nullptr;
  outer_neighflag = legacy_lists ? 1 : 0;
  flux_enable = 0;
  site_kernel = 1;
  site_need_lists = 1;
  type_map = nullptr;
  cutmax = 0.0;
  centers_ncalls = -1;
}

/* ---------------------------------------------------------------------- */

PairCACNEP::~PairCACNEP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete[] type_map;    // map is freed by Pair::~Pair()
  }
}

/* ---------------------------------------------------------------------- */

void PairCACNEP::allocate()
{
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  map = new int[n + 1];
  type_map = new int[n + 1];
  max_nodes_per_element = atom->nodes_per_element;
  PairCAC::allocate();
}

/* ----------------------------------------------------------------------
   pair_style cac/nep [one]
------------------------------------------------------------------------- */

void PairCACNEP::settings(int narg, char **arg)
{
  PairCAC::settings(narg, arg);
}

/* ----------------------------------------------------------------------
   pair_coeff * * model.txt Element1 Element2 ...
------------------------------------------------------------------------- */

void PairCACNEP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != 3 + atom->ntypes)
    error->all(FLERR, "Incorrect args for pair_coeff cac/nep: * * model.txt <element per type>");

  map_element2type(narg - 3, arg + 3);
  for (int i = 0; i <= atom->ntypes; ++i) type_map[i] = map[i];

  nep_model.init_from_file(utils::get_potential_file_path(arg[2]), comm->me == 0);
  nep_model.update_type_map(atom->ntypes, type_map, elements);

  if (nep_model.paramb.model_type != 0)
    error->all(FLERR, "pair_style cac/nep supports NEP potential models only");

  cutmax = std::max(nep_model.paramb.rc_radial_max, nep_model.paramb.rc_angular_max);
  cut_global_s = cutmax;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = 1; j <= atom->ntypes; j++) cutsq[i][j] = cutmax * cutmax;
}

/* ---------------------------------------------------------------------- */

void PairCACNEP::init_style()
{
  PairCAC::init_style();
  atom->max_neigh_inner_init = maxneigh_quad_inner = 10;
  atom->max_neigh_outer_init = maxneigh_quad_outer = 500;
  atom->outer_neigh_flag = legacy_lists ? 1 : 0;
  atom->inner_only_flag = legacy_lists ? 0 : 2;
  site_enumerate = legacy_lists ? 0 : 1;
  if (atom->cac_flux_flag) error->all(FLERR, "pair_style cac/nep does not support CAC flux output");
}

/* ----------------------------------------------------------------------
   the quadrature lists must reach 2*rc + skin so that every neighbor of a
   quadrature point has its own complete neighborhood
------------------------------------------------------------------------- */

double PairCACNEP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  site_list_cut = cutmax + neighbor->skin;
  return 2.0 * cutmax;
}

/* ---------------------------------------------------------------------- */

void PairCACNEP::force_densities(int, double, double, double, double, double &, double &,
                                 double &)
{
  error->all(FLERR, "pair_style cac/nep requires the CAC site-table path; "
             "sector, flux and ghost-quadrature options are not supported");
}

/* ----------------------------------------------------------------------
   centers = sites that are quadrature points or appear in an inner list
   (within rc + skin of a quadrature point at the last neighbor build)
------------------------------------------------------------------------- */

void PairCACNEP::build_centers()
{
  centers.clear();
  site_numneigh.assign(nsite, 0);
  site_firstneigh.assign(nsite, nullptr);
  for (int s = 0; s < nsite; s++) {
    if (!site_has_list[s]) continue;
    centers.push_back(s);
    site_numneigh[s] = slist_first[s + 1] - slist_first[s];
    site_firstneigh[s] = slist.data() + slist_first[s];
  }
  centers_ncalls = site_table_ncalls;
}

/* ---------------------------------------------------------------------- */

void PairCACNEP::compute_sites(int /*eflag*/, int /*vflag*/)
{
  if (centers_ncalls != site_table_ncalls || (int) site_numneigh.size() != nsite) build_centers();

  const int cac_virial = atom->CAC_virial;
  site_f.assign(3 * (size_t) nsite, 0.0);
  site_e.assign(nsite, 0.0);
  site_xptr.resize(nsite);
  site_fptr.resize(nsite);
  for (int s = 0; s < nsite; s++) {
    site_xptr[s] = &site_x[3 * (size_t) s];
    site_fptr[s] = &site_f[3 * (size_t) s];
  }
  double **vptr = nullptr;
  if (cac_virial) {
    site_v.assign(9 * (size_t) nsite, 0.0);
    site_vptr.resize(nsite);
    for (int s = 0; s < nsite; s++) site_vptr[s] = &site_v[9 * (size_t) s];
    vptr = site_vptr.data();
  }

  double total_potential = 0.0;
  double total_virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  nep_model.compute_for_lammps(nsite, (int) centers.size(), centers.data(), site_numneigh.data(),
                               site_firstneigh.data(), site_type.data(), type_map,
                               site_xptr.data(), total_potential, total_virial, site_e.data(),
                               site_fptr.data(), vptr);

  for (int pq = 0; pq < nquadpt; pq++) {
    const int s = quad_site[pq];
    quad_force[3 * (size_t) pq] = site_f[3 * (size_t) s];
    quad_force[3 * (size_t) pq + 1] = site_f[3 * (size_t) s + 1];
    quad_force[3 * (size_t) pq + 2] = site_f[3 * (size_t) s + 2];
    quad_energy[pq] = site_e[s];
    if (cac_virial)
      for (int d = 0; d < 6; d++) quad_virial[6 * (size_t) pq + d] = site_v[9 * (size_t) s + d];
  }
}

#endif
