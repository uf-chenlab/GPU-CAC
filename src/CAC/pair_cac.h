/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(cac,PairCAC)

#else

#ifndef LMP_PAIR_CAC_H
#define LMP_PAIR_CAC_H

#include "pair.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

template <class DeviceType> class CACSiteDevice;    // Kokkos device side of the site table

class PairCAC : public Pair {
  template <class D> friend class CACSiteDevice;
 public:
	double cutmax;                // max cutoff for all elements
  int pre_force_flag;           // set to 1 if computing something before force   
   
  PairCAC(class LAMMPS *);
  virtual ~PairCAC();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **){}
  virtual void init_style();
  virtual double init_one(int, int){ return 0.0; }

  //functions for Asa_Data and class to obtain
  double shape_function(double, double, double,int,int);
  double shape_function_derivative(double, double, double,int,int,int);
 
  //set of shape functions
  double quad_shape_one(double s, double t, double w){ return (1-s)*(1-t)*(1-w)/8;}
  double quad_shape_two(double s, double t, double w){ return (1+s)*(1-t)*(1-w)/8;}
  double quad_shape_three(double s, double t, double w){ return (1+s)*(1+t)*(1-w)/8;}
  double quad_shape_four(double s, double t, double w){ return (1-s)*(1+t)*(1-w)/8;}
  double quad_shape_five(double s, double t, double w){ return (1-s)*(1-t)*(1+w)/8;}
  double quad_shape_six(double s, double t, double w){ return (1+s)*(1-t)*(1+w)/8;}
  double quad_shape_seven(double s, double t, double w){ return (1+s)*(1+t)*(1+w)/8;}
  double quad_shape_eight(double s, double t, double w){ return (1-s)*(1+t)*(1+w)/8;}
 
  virtual double memory_usage();

 protected:
  int outer_neighflag, current_element_index;
  int sector_flag, flux_enable;
  int ghost_quad;
  double cutforcesq;
  double **scale;
  int *current_element_scale;
  double mapped_volume;
  int reneighbor_time;
  int max_nodes_per_element, neigh_poly_count;
  double virial_density[6], flux_density[24];
  int ***flux_neigh_intercepts, *flux_neigh_nintercepts, **flux_neigh_indices, **nplane_intersects;
  class FixCACAllocVector *fix_cac_alloc_vector;  // fix that stores nodal flux information

  //stores quadrature point coordinates and calculation coefficients
  double **quadrature_point_data;
  int quadrature_point_max;
  int *quadrature_counts, *e2quad_index;
  int max_quad_per_element;
  double *quadrature_weights;
  double *quadrature_abcissae;
  int quadrature_node_count;
	
  double cut_global_s;
  double cutoff_skin;
  int quad_allocated;
  int one_layer_flag;
  //used to call set of shape functions
  typedef double(PairCAC::*Shape_Functions)(double s, double t, double w);
  Shape_Functions *shape_functions;

  double element_energy;
  int quad_eflag, flux_compute, quad_flux_flag;
  double quadrature_energy;
  double **mass_matrix;
  double **mass_copy;
  double **force_column, **current_virial_projection, **current_flux_projection;
  double *current_nodal_forces, current_position[3], current_velocity[3];
  double *current_force_column, *current_virial_column, *current_flux_column;
  double *current_x;
  int   *pivot;
  double **shape_quad_result;
  double **current_nodal_positions;
  double **neighbor_copy_ucell;
  int **neighbor_copy_index;
  int neighbor_element_type;
  int old_atom_count,old_all_atom_count, old_quad_count;
  int *old_atom_etype, *old_all_atom_etype;
  int ***inner_quad_lists_index, ***outer_quad_lists_index, ***add_quad_lists_index;
  double ***inner_quad_lists_ucell, ***outer_quad_lists_ucell, ***add_quad_lists_ucell;
  int *inner_quad_lists_counts, *outer_quad_lists_counts, *add_quad_lists_counts;
  int atomic_flag;
  int nmax, nmax_surf;
  int neighrefresh;
  int maxneigh;
  int maxneigh_quad_inner, maxneigh_quad_outer;
  int maxneigh2;
  int current_element_type, current_poly_count;
  int natomic, atomic_counter;
  int *type_array, type_map, *type_maps;
  int poly_counter;
  int qi, pqi;
  int local_inner_max, local_outer_max, local_add_max, local_all_max;
  int vel_inner_max, vel_outer_max;
  double **inner_neighbor_coords, **inner_neighbor_velocities;
  double **outer_neighbor_coords, **outer_neighbor_velocities;
  double **add_neighbor_coords, **add_neighbor_velocities;
  int *inner_neighbor_types, *outer_neighbor_types, *add_neighbor_types;
  double *inner_neighbor_charges, *outer_neighbor_charges, *add_neighbor_charges;

	virtual void allocate();

  //further CAC functions 
  void check_existence_flags();
  void compute_mass_matrix();
  void compute_forcev(int);
  void set_shape_functions();
  void interpolation(int, double, double, double);
  void allocate_quad_memory();
  void init_quad_arrays();
  void LUPSolve(double **A, int *P, double *b, int N, double *x);
  void mass_solve(double *b, int n, double *x);
  int LUPDecompose(double **A, int N, double Tol, int *P);
  double shape_product(int,int);
  void quadrature_init(int degree);
  virtual void compute_intersections();
  virtual void quad_neigh_flux();
  virtual void current_quad_flux(int, double, double, double);
  virtual double pair_interaction(double, int , int){ return 0.0; }
  virtual double pair_interaction_q(double, int, int, double, double){ return 0.0; }
  virtual void pre_force_densities() {}
  virtual void force_densities(int, double, double, double, double, double
    &fx, double &fy, double &fz) {}
  int mldivide3(const double mat[3][3], const double *vec, double *ans);

  // ---------------------------------------------------------------------
  // site table (performance layer, built once per neighbor list build)
  //   every quadrature point and every virtual/real neighbor in the
  //   quadrature neighbor lists is a lattice "site" (entity, poly, cell)
  //   or a real atom; sites are de-duplicated so their positions are
  //   interpolated once per step instead of once per quadrature point
  // ---------------------------------------------------------------------

  int site_table_enable;         // 0 = original code path (pair_modify-like switch)
  int site_kernel;               // 1 if derived style implements compute_sites()
  // CAC_TIMING=1: accumulated wall time (s) of the site-path phases, printed at exit by rank 0
  //  0 build_site_table  1 update_site_positions (rebuild)  2 build_site_lists  3 compute_sites
  //  4 nodal projection  5 positions (per step)  6 kk upload  7 kk node pack+h2d+interp
  //  8 kk NEP kernels  9 kk gather+d2h+host copy
  double tm[16];
  int tm_on;
  // cell grid over the site positions (cells of ~site_list_cut/2, sites sorted by cell)
  struct SiteGrid {
    double lo[3], binsize[3];
    int nb[3], reach[3];
    std::vector<int> sbin, bstart, order;    // cell of each site, first sorted slot of each cell, site id by sorted slot
    std::vector<double> xb;                  // coordinates in sorted order
  };
  void build_site_grid(SiteGrid &g);
  void build_site_table_fast();    // site-only mode: sites numbered per element, built in parallel
  virtual int build_lists_on_host() { return 1; }    // 0: the pair style builds the site lists itself
  int site_sort_lists;           // 1: sort every site neighbor list by site id (helps gather-heavy kernels such as EAM)
  int site_legacy;               // CAC_LEGACY_LISTS set: original quadrature-list scan instead of site-only lists
  int site_center_lists;         // 1: sites around the quadrature sites also get neighbor lists (many-body)
  // neighbor list of quadrature point pq (inner list, or the site list of its site in site-only mode)
  inline const int *quad_list(int pq, int &n) const
  {
    if (site_only_lists) {
      const int s = quad_site[pq];
      n = slist_first[s + 1] - slist_first[s];
      return slist.data() + slist_first[s];
    }
    n = inner_first[pq + 1] - inner_first[pq];
    return inner_site.data() + inner_first[pq];
  }
  // threaded "center + scatter" support: forces/energies/virials of every site accumulated per thread
  std::vector<std::vector<double>> sc_f, sc_e, sc_v;
  std::vector<int> sc_centers;
  bigint sc_centers_ncalls;
  int sc_centers_nsite;
  int scatter_begin(int need_e, int need_v);     // returns the number of threads; builds the center list
  void scatter_finish(int need_e, int need_v);   // reduce into quad_force / quad_energy / quad_virial
  FILE *dump_fp;                 // CAC_DUMP_QUAD: per-quadrature-point dump written by the original (non-site) path
  int site_host_positions;       // 0: the pair style keeps site positions on the device (host copy only at rebuilds)
  int site_only_lists;           // 1: quadrature lists come from the site table, not from NPairCAC scans
  int site_enumerate;            // 1: site table also holds every lattice site of all local+ghost elements
  int site_need_lists;           // 1 if compute_sites() needs neighbor lists of non-quad sites
  double site_list_cut;          // cutoff for site neighbor lists (many-body)
  bigint site_table_ncalls;      // neighbor->ncalls when the table was last built
  int site_table_valid;
  int site_table_reported;
  int use_sites;                 // 1 while compute() gathers from the site table

  int nsite;                           // # of unique sites (local + ghost owners)
  std::vector<int> site_owner;         // atom/element index owning the site
  std::vector<int> site_poly;          // poly (internal DOF) index
  std::vector<int> site_type;          // atom type of the site
  std::vector<double> site_q;          // charge of the site (charge styles)
  std::vector<double> site_w;          // 8 Q8 shape weights per site (element sites)
  std::vector<double> site_x;          // current positions, 3 per site
  std::unordered_map<uint64_t, int> site_map;    // fallback for very large elements
  std::vector<int> atom_site;                    // site id of each real atom (local + ghost)
  std::vector<long> cell_base;                   // per (owner,poly): offset into cell_site, -1 unset, -2 hashed
  std::vector<int> cell_site;                    // site id of every cell of (owner,poly)

  int nquadpt;                         // # of local quadrature evaluation points (pqi)
  std::vector<int> quad_site;          // site id of each quadrature evaluation point
  std::vector<int> quad_owner;         // owning local atom/element
  std::vector<int> quad_qi;            // index into quadrature_point_data
  std::vector<int> inner_first, inner_site;    // CSR: inner list entries as site ids
  std::vector<int> outer_first, outer_site;    // CSR: outer list entries as site ids
  std::vector<double> quad_shape;      // 8 projection weights per pqi at (sq,tq,wq)

  std::vector<int> slist_first, slist;    // CSR site neighbor lists (many-body, site_need_lists)
  std::vector<int> site_has_list;

  std::vector<double> quad_force;      // force density per quadrature point (3)
  std::vector<double> quad_energy;     // quadrature energy per quadrature point
  std::vector<double> quad_virial;     // virial density per quadrature point (6)

  double Minv_q8[8][8];                // inverse of the (constant) Q8 mass matrix
  int minv_ready;

  uint64_t site_key(int owner, int poly, const double *ucell);
  int site_lookup(int owner, int poly, const double *ucell);
  void build_site_table();
  void build_site_lists();
  void update_site_positions();
  void compute_minv();
  void compute_site_path(int, int);
  virtual void compute_sites(int, int) {}
  void interpolation_from_sites(int);
  void init_quad_arrays_from_sites();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: LU matrix is degenerate

For some reason the mass matrix for one of the finite elements defined
in your CAC model is ill conditioned.

E: Unexpected argument in CAC pair style invocation

Self-explanatory.  Check the input script or data file. See the documention
for appropriate syntax.

E: CAC Pair style requires a CAC atom style

Self-explanatory.

E: CAC Pair styles cannot be invoked with hybrid; also 
don't use the word hybrid in your style name to avoid this error

Self-explanatory.

E: CAC Pair style requires a CAC comm style

Self-explanatory.

E: minimum points exceed element domain

The minimization algorithm used to locate adjacent finite element 
geometry and complete the virtual atom neighborhood is encountering
an invalid solution. Request help for this supposedly impossible scenario.
See the documentation for contact info.

*/
