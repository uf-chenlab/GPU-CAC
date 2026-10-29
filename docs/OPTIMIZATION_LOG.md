# CAC efficiency audit and optimisation log

Started 2026-10-26. Updated after every step (change, validation, benchmark). Meeting summary at the bottom.
Hardware/env: HiPerGator, phillpot account; GPU = NVIDIA L4 (hpg-turin), env of `proj/new_workflow/gpuq/env.sh` (see STATUS.md item 6). Build: `tools/build_cuda.sbatch`.

## Benchmark definitions
- **B1 (NEP, mixed AlN)**: `test/gpu_bench/in.bench_cacnep`, 8272 atom-equivalent nodes/atoms, 49152 quadrature points, 49152 unique sites, 87.9M site-list entries, 20 steps NVE. Metric: Loop time (s) and Total wall time (s, includes setup).
- **V1 (validation)**: `test/gpu_run/in.cacnep_v`, 3336 quadrature points vs full-atom NEP (`tools/compare_quad.py`). Must stay at <=1e-12 (fp64) force error.
- **Regression suite**: `bench/run_suite.sh` + `compare_suite.py` (EAM, LJ, Tersoff, SW, MPI) must remain bit-identical / 1e-16 to the original.

## Baseline (before this audit), 2026-10-26
| Case | Loop (s) | Pair (s) | Total wall (s) |
|---|---|---|---|
| CPU cac/nep, 1 rank | 56.63 | 56.56 | 71 |
| GPU cac/nep/kk fp64, L4 | 0.404 | 0.383 | 13 |
| GPU cac/nep/kk df64, L4 | 0.257 | - | - |
| GPU cac/nep/kk fp32, L4 | 0.132 | 0.099 | 13 |
| Kokkos OpenMP 8 thr (earlier) | 5.34 | - | - |

Observations:
- Per step, the loop is pair-dominated. Wall time is dominated by **setup** (13 s total vs 0.4 s loop on GPU): neighbour build and site table (87.9M list entries).
- CPU cost is the NEP kernel itself, per unique site (about 1.15 ms/site/step); CAC-specific overhead is negligible on CPU in this case.

## Steps
(see below, appended as work proceeds)

### Step 0: profile (perf, B1, GPU run, fp64; job 47837943)
`run 0` (setup only) and 20 steps both take 13 s wall; the 20 MD steps themselves are 0.4 s. So **the neighbour-list build dominates**, and it runs at every reneighbouring (one rebuild = about 650 GPU steps).
Flat profile (setup phase):
| Share | Function |
|---|---|
| 43.2% | NPairCAC::neighbor_accumulate |
| ~15% | ASA nonlinear solver (mygrad_surfmin, myvalue_surfmin, asa_cg, asa_descent, asa_grad_proj ...) |
| 8.0% | NPairCAC::shape_function / shape_function_derivative |
| 5.8% | PairCAC::site_lookup |
| 4.8% | NPairCAC::build |
| 3.8% | NPairCAC::quad_list_build |
| 3.6% | llroundf32x (rounding in site_lookup / cell index) |
| 2.8% | PairCACNEPKokkos::upload_site_table |
| ~4% | malloc/realloc/free |
Conclusion: optimise NPairCAC (neighbour build) first; then site-table construction and upload; then per-step kernels.

## Model set (all 49152 atom-equivalents, AlN wurtzite, box 78x92x85 A, `models/`, `tools/make_cac_model.py`)
| Model | Content |
|---|---|
| atoms | 49152 atoms, 0 elements |
| elem4 | 96 elements (4x4x4-cell blocks, 512 sites x 8 poly... 4096 sites each) |
| elem8 | 12 elements (8x8x8-cell blocks) |
| mix4  | 8192 atoms + 80 elements (the old B1 benchmark) |
Sweep script: `tools/bench_models.sbatch` (run in the interactive allocation with `srun --jobid=<id> --overlap`). `static` = 20 steps with no rebuild; `rebuild` = 5 steps with a neighbour rebuild at every step (`neigh_modify every 1 delay 0 check no`), so `Neigh` / 5 = cost of the LAMMPS neighbour build per rebuild and (`Pair` - 5 x static step) / 5 = extra pair-style work per rebuild.
Machine: GPU = L4 (hpg-turin), fp64 `cac/nep/kk`.

### Step 1: opt1 + opt2 = inner-only quadrature lists, then site-only lists
Findings: the profile of the 13 s setup was dominated by NPairCAC (per-quadrature-point scan of every neighbouring element in isoparametric space, with an ASA nonlinear solve per element pair and shape-function evaluation per cell). Every quadrature point is already snapped to a lattice site (`quadrature_point_data[0..2]` = lattice-snapped s,t,w), so all information the pair style needs can be generated from site positions alone.
Changes (cac/nep and cac/nep/kk only; env `CAC_LEGACY_LISTS=1` restores the old path):
1. `atom->inner_only_flag` (new): 1 = quadrature list cutoff is rc+skin and there is no outer list; 2 = **site-only mode**: `NPairCAC::build` skips the whole per-quadrature-point list build (`quad_list_build`, `neighbor_accumulate`, ASA).
2. `PairCAC::build_site_table`: with `site_enumerate`, every lattice site of every local+ghost element is registered directly (`site_lookup` on all cell centres) instead of being discovered through the 2rc+skin outer lists.
3. `PairCAC::build_site_lists`: in site-only mode, the quadrature-site neighbour lists and the centre lists are built from the same cell list over site positions (new `query` lambda); no inner/outer entries are stored (list entries 87.9M -> 0).
Validation (V1, fp64, vs full-atom NEP): max force error 4.3e-13 eV/A (legacy path 4.5e-13), 9720 sites in both.
Results (fp64 L4, times in s):
| Model | Case | Legacy wall | New wall | Legacy Neigh/rebuild | New Neigh/rebuild |
|---|---|---|---|---|---|
| mix4 | static 20 steps | 12 | **1** | - | - |
| mix4 | rebuild each step | 63 | **6** | 8.62 | **0.27** |
| elem4 | static | 14 | **1** | - | - |
| elem4 | rebuild | 76 | **4** | 10.59 | **0.002** |
| elem8 | static | 2 | 1 | - | - |
| elem8 | rebuild | 8 | 4 | 0.44 | **0.0001** |
| atoms | static | 7 | **3** | - | - |
| atoms | rebuild | 24 | 12 | 2.78 | **1.12** |
Per-step (static) loop times are unchanged (0.37-0.44 s / 20 steps = ~19-22 ms/step).
Remaining after this step: pair-side cost per rebuild ~0.74 s (site table, cell lists, host->device upload) vs 19 ms per step; the atoms-only model still pays 1.1 s per rebuild in the LAMMPS neighbour code.

### Step 2: pair-side rebuild path (OpenMP + sorted cell list + device expansion), atom-list skip
Profile after step 1 (rebuild every step, perf, elem4/atoms): `upload_site_table` 46%/17%, `build_site_lists` 45%/16%, and for the atoms-only model `NPairCAC::build` 57% (it builds 18 A atom-atom neighbour lists of 49152 atoms that nothing reads in site-only mode).
Changes:
1. `NPairCAC::build`: in site-only mode atoms skip the element/atom neighbour scan (only element lists are used, for the interface-resolution flags).
2. `PairCAC::build_site_lists` rewritten: counting sort of the sites into cells of ~rc/2 with contiguous coordinate arrays, one contiguous range per x-row of cells, threaded with OpenMP (chunks of sites per thread, concatenated in site order so results do not depend on the thread count); BUILD_OMP=on added to the build (`tools/rebuild.sh reconf`); runs use `mpirun --bind-to none` and OMP_NUM_THREADS=16.
3. `PairCACNEPKokkos::upload_site_table`: static site arrays (node rows, weights, types, quadrature map) are cached and only re-uploaded when the site set changes; the neighbour lists go up as flat CSR and are expanded to the padded 2d array by a device kernel (`CACNepExpand`); no zero-fill of the padded array, host fill loop and staging copy removed.
Validation: V1 fp64 4.26e-13 eV/A (unchanged).
Results (fp64, L4; "rebuild" = 5 steps with a neighbour rebuild every step):
| Model | rebuild loop before (s) | after (s) | Neigh before (s, 5 rebuilds) | after |
|---|---|---|---|---|
| atoms | 9.38 | **0.64** | 5.60 | **0.007** |
| elem4 | 3.82 | **0.50** | 0.009 | 0.009 |
| elem8 | 3.73 | **0.45** | 0.0003 | 0.0003 |
| mix4 | 5.16 | **0.69** | 1.36 | **0.158** |
Static 20-step loops unchanged (0.37-0.43 s on the GPU). Per rebuild, the pair-side overhead drops from ~0.74 s to ~0.09 s (about 4-5 GPU steps).

### Step 3: CPU NEP (`cac/nep`, also `pair_style nep` via NEP3::compute_for_lammps): fused + threaded
Observation: the CPU NEP of the original port loops over the whole inner list (~340 neighbours) three times per centre (descriptor, radial force, angular force), tests the cutoff each time, recomputes the radial basis for every angular order n (5x), and has no threading.
Change (`src/NEP/nep.cpp`, `compute_for_lammps_fast`; `NEP_CPU_LEGACY=1` selects the old routines): per centre, neighbours are classified once into radial (<8 A), angular (<4 A) and ZBL (<2.5 A) lists; the radial basis, its derivative, g_n and its derivative are evaluated once per (centre, neighbour); the network is applied; both force passes reuse the stored values; expansion coefficients are reordered per type pair; centres are distributed over OpenMP threads (dynamic chunks of 16) with thread-private force/virial buffers reduced in parallel.
Validation (V1, vs full-atom NEP): max force error 4.15e-13 eV/A (legacy routine: 4.15e-13).
Results (mix4, `cac/nep one`, per MD step, CPUs of an L4 node):
| Threads | s/step | Speed-up vs original (2.83 s) |
|---|---|---|
| 1 (fused) | 1.49 | 1.9x |
| 16 | 0.105 | 27x |

## Part B: classical potentials (EAM, LJ, SW, Tersoff), CPU
Scope reminder from the user: the normal potentials matter more than NEP-CPU. Benchmarks: `tools/bench_classic.sh` (input `test_classic/in.classic`, models in `models_classic/`: Cu fcc 16384 atom-equivalents and Si diamond 13824 atom-equivalents, each as all-atoms / all small elements (4-cell blocks, `elem4`) / all large elements (`elem8` Cu, `elem6` Si) / mixed). `static` = 5 steps, no rebuild; `rebuild` = 3 steps with a neighbour rebuild at every step, so `Neigh`/3 is the cost of one LAMMPS-level rebuild. 1 MPI rank, 1 thread, current build before any classical-potential change (the NEP changes of parts 1-3 do not touch these styles).

### Baseline (CPU, s)
| style | model | static loop (5 steps) | rebuild loop (3 steps) | Neigh in rebuild run |
|---|---|---|---|---|
| eam | atoms | 0.154 | 1.10 | 0.886 |
| eam | elem4 | 0.081 | 3.75 | 3.27 |
| eam | elem8 | 0.034 | 0.221 | 0.145 |
| eam | mix4 | 0.111 | 3.49 | 3.04 |
| eam/alloy | elem4 | 0.112 | 4.93 | 4.35 |
| eam/fs | elem4 | 0.279 | 7.39 | 6.17 |
| lj | elem4 | 0.021 | 1.21 | 1.10 |
| sw | atoms | 0.148 | 0.391 | 0.085 (pair 0.14 per 5 steps) |
| sw | elem4 | 0.226 | 1.85 | 1.72 |
| tersoff | atoms | 0.440 | 0.468 | |
| tersoff | elem4 | 0.500 | 1.69 | 1.40 |
Same pattern as NEP: a time step is cheap (10-100 ms), but one rebuild of the CAC quadrature neighbour lists costs 0.3-2.4 s, i.e. 30-150 steps. Per-step cost: Tersoff ~100 ms/step for 13824 atoms (7 us/atom) and SW 30-45 ms/step: both still use the per-quadrature-point path that recomputes everything from long inner+outer lists.
Plan: (1) site-only lists for EAM/LJ (+ OpenMP site kernels), (2) site-based centre kernels for SW and Tersoff (centres + scattered forces, threaded), (3) GPU (Kokkos) versions of the classical styles on top of the device site/list machinery built for NEP.

### Step B1: site-only lists for EAM (eam, eam/alloy, eam/fs) and LJ
Same mechanism as for NEP: no per-quadrature-point scan in `NPairCAC` (flag `atom->inner_only_flag`: 2 = site-only with halved list cutoff for the many-body EAM, 3 = site-only single range for LJ); all lattice sites of local+ghost elements are enumerated into the site table; quadrature-site lists (and, for EAM, the centre lists around them) come from the sorted cell grid, threaded. The EAM/LJ kernels read the quadrature lists through `PairCAC::quad_list()`. `CAC_LEGACY_LISTS=1` restores the original scan. Non-site paths (flux, sectors, ghost quadrature) require the legacy lists and say so.
Regression suite (`bench/run_suite.sh`, 10 cases incl. 4 MPI ranks and CG minimisation) vs the original USER-CAC results: all OK, thermo max rel diff <= 4.1e-16, dumps identical (eam, eam/alloy, eam/fs, lj, sw, tersoff, eam_np4, lj_np4, eam_min, eam_full1).
Benchmarks (1 rank, 1 thread; loop time s; static = 5 steps, rebuild = 3 steps with a rebuild every step):
| style | model | static before | static after | rebuild before | rebuild after | speed-up (rebuild run) |
|---|---|---|---|---|---|---|
| eam | atoms | 0.154 | 0.089 | 1.10 | 0.075 | 15x |
| eam | elem4 | 0.081 | 0.069 | 3.75 | 0.055 | 69x |
| eam | elem8 | 0.034 | 0.023 | 0.221 | 0.021 | 11x |
| eam | mix4 | 0.111 | 0.071 | 3.49 | 0.122 | 29x |
| eam/alloy | elem4 | 0.112 | 0.168 | 4.93 | 0.123 | 40x |
| eam/fs | elem4 | 0.279 | 0.398 | 7.39 | 0.279 | 26x |
| lj | atoms | 0.023 | 0.026 | 0.224 | 0.028 | 8x |
| lj | elem4 | 0.021 | 0.018 | 1.21 | 0.022 | 55x |
| lj | mix4 | 0.028 | 0.018 | 1.21 | 0.163 | 7x |
(static timings vary by +-20% between runs; the Cu_mishin/fs static loops are noise-level different.) The rebuild of the mixed (atom+element) models still costs 20-50 ms in `NPairCAC::build` (to be profiled).

### Step B2: site kernels for SW and Tersoff (centre + scatter, threaded)
Both styles had no site kernel: every quadrature point rebuilt cluster neighbour lists from its inner+outer lists and re-evaluated every three-body term that contains it (about 3x redundant work, plus the same slow list scan as above).
New `compute_sites()` for `cac/sw` and `cac/tersoff` (shared helper `PairCAC::scatter_begin/scatter_finish`): every centre (a quadrature site or a site within the force cutoff of one; lists from the sorted cell grid) evaluates its own terms once with the LAMMPS formulas and scatters the forces onto all participants into thread-private buffers; the force density of a quadrature point is the sum of what it receives. SW: pair terms once per pair (smaller site index), three-body terms once per (centre, j<k). Tersoff: repulsive pair once per pair; for every ordered pair (i,j) the zeta sum, `force_zeta` and `attractive` on i, j, k. Energies and virials are split as in the original: pair terms 1/2 to each end, three-body virial 1/3 to each participant. OpenMP over centres (dynamic chunks of 16). `CAC_LEGACY_LISTS=1` selects the original per-quadrature-point code.
**Validation against full-atom LAMMPS** (`tools/validate_classic.sh`, `tools/force_check.sh`; Si diamond / Cu fcc models with 0.04 A jitter, all-atom, all-element and mixed, element scales 4 and 3; `expand_cac.py` gives the equivalent atomistic configuration):
- total energy: CAC (legacy and new) vs `pair_style sw/tersoff/eam/eam/alloy/lj/cut`: agree to 1e-14 relative in every case (e.g. si_a sw -55737.387785807652 vs -55737.387785807208);
- per-quadrature-point force and site energy of the new kernels vs full-atom forces: max |dF| <= 9e-13 relative (<= 6e-13 eV/A absolute) for sw, tersoff, eam, eam/fs, lj in all-atom, all-element, mixed and odd-scale models.
**Finding:** the original per-quadrature-point SW/Tersoff code gives a wrong total energy (and slightly different forces) for models that mix atoms with scale-3 elements: -21423 eV instead of -56280 eV (suite model `si_small_jit`, same reference in `bench/suite_orig`). The new kernels give the full-atom value. Consequently `sw_s1`/`tersoff_s1` of the old regression suite (references taken from the original code) now differ by design; their reference should be the full-atom run above.
Benchmarks (1 rank; loop time in s; static = 5 steps, rebuild = 3 steps with a rebuild every step; base = original ported code, t1/t16 = new code with 1/16 OpenMP threads; EAM/LJ kernels are not yet threaded here):
| style | model | static base | static t1 | static t16 | rebuild base | rebuild t1 | rebuild t16 |
|---|---|---|---|---|---|---|---|
| sw | atoms | 0.148 | 0.041 | 0.0085 | 0.391 | 0.069 | 0.017 |
| sw | elem4 | 0.226 | 0.018 | 0.0038 | 1.854 | 0.047 | 0.0093 |
| sw | elem8 | 0.058 | 0.014 | 0.0025 | 0.337 | 0.041 | 0.0064 |
| sw | mix4 | 0.241 | 0.019 | 0.0041 | 1.856 | 0.112 | 0.077 |
| tersoff | atoms | 0.440 | 0.099 | 0.0119 | 0.468 | 0.080 | 0.016 |
| tersoff | elem4 | 0.500 | 0.087 | 0.0086 | 1.691 | 0.068 | 0.0105 |
| tersoff | elem8 | 0.121 | 0.061 | 0.0054 | 0.368 | 0.046 | 0.0068 |
| tersoff | mix4 | 0.496 | 0.085 | 0.0088 | 1.805 | 0.164 | 0.107 |
SW per step: elem4 45 ms -> 0.8 ms (59x); Tersoff elem4 100 ms -> 1.7 ms (58x); a Tersoff rebuild run on elem4 is 160x faster than before.

## Part C: verification (added at the user's request: compare per-atom forces, energies and virials, not only totals; identical results; multi-core)
Tools: `tools/verify_all.sh` (matrix), `tools/verify_pq.py` (point-by-point comparison: max abs/rel deviation, number of bit-identical values, max difference in ulp), `tools/force_check.sh`, `tools/validate_classic.sh`, `CAC_DUMP_QUAD=<prefix>` (writes, per MPI rank, for every quadrature point: index, type, position, force density, energy, virial; implemented in BOTH the original per-point path and the new site path so the two can be compared point by point).
Checks per style x model (all-atom `a`, all-element `e`, mixed `m`, mixed with odd element scale `s3`; jittered 0.04 A) x MPI ranks x OpenMP threads:
- A: original code path (CAC_LEGACY_LISTS=1) vs new path on the same decomposition, point by point (force density, energy, virial), nodal forces (text dump, 6 digits, identical to the byte), thermo (PE, temperature, pressure);
- B: new path with N ranks / T threads vs new path on 1 rank / 1 thread, points matched by position;
- C: new path vs full-atom LAMMPS (`sw`, `tersoff`, `eam`, `eam/alloy`, `eam/fs`, `lj/cut`) on the equivalent atomistic configuration (`expand_cac.py`): force and energy of every atom; C2: global virial pressure;
- D: 10 MD steps with a neighbour rebuild at every step, original vs new.
What "identical" can mean: the new kernels sum the same terms in a different order (cell-grid neighbour order, scatter instead of gather, threads), so floating-point sums differ in the last bits. Measured deviations are reported as max relative deviation and in ulps; for a fixed rank and thread count the result is deterministic (static schedule).

### Bugs found by these checks
1. (mine, fixed) Site-only mode cleared `outer_neigh_flag`, which `compute_surface_depths` used to size the explicitly sampled surface layer (2*rc instead of rc): quadrature-point sets and weights differed from the original, so nodal forces and dynamics deviated by ~1e-5-1e-2 even though static energies and per-atom forces matched. Fixed (surface depth now depends on the list mode); nodal forces are now byte-identical to the original. **All earlier NEP timings for models with elements were measured with the shallower layer and are re-measured below.**
2. (original code) `cac/sw` and `cac/tersoff` original path: wrong total energy for mixed atom+element models with odd element scale (e.g. -21423 eV instead of -56280 eV); SW global virial pressure is 0 for atom-only models (full-atom LAMMPS: 2981.1089720970549 bar; new path 2981.1089720970594 bar, rel 1.5e-15).
3. (original code) the original non-site path segfaults in `PairCAC::LUPSolve` (virial projection) on MPI ranks whose first local element is an atom when the virial is requested (uninitialised LU pivot array).

### Verification results, classical potentials (CPU; 1,2,4,8 MPI ranks; 1 and 4 OpenMP threads; models a/e/m/s3; jitter 0.04 A)
Counts: sw 92/92, tersoff 92/92, eam 105/105, eam/alloy 105/105, eam/fs 105/105, lj 105/105 comparisons passed (`logs/verify_full_*.log`). Worst-case deviations (relative to the largest value of each quantity; 13824 / 6912 quadrature points per model; values of different runs are *not* bit-identical because sums are formed in a different order, see below):
| check | style | force density | energy | virial | thermo (PE, T, press) |
|---|---|---|---|---|---|
| A original code vs new, same ranks | sw | 4.6e-16 | 8.2e-16 | 4.7e-16 | 4.0e-16 |
| | tersoff | 3.7e-14 | 1.1e-15 | (not comparable, see below) | 4.1e-16 (PE,T) |
| | eam / alloy / fs | 1.2e-13 / 1.3e-13 / 1.2e-13 | 2.4e-15 / 2.6e-15 / 2.1e-15 | 5.3e-14 / 4.4e-14 / 5.8e-14 | 1.4e-14 / 1.3e-14 / 4.9e-14 |
| | lj | 1.0e-13 | 3.3e-15 | 1.0e-13 | 1.3e-14 |
| B new on N ranks / T threads vs 1 rank / 1 thread | sw | 1.4e-14 | 1.0e-15 | 1.3e-14 | 6.4e-15 |
| | tersoff | 8.7e-15 | 3.6e-16 | 8.0e-15 | 8.3e-15 |
| | eam / alloy / fs | 7.9e-16 / 5.2e-16 / 5.0e-16 | 1.4e-15 / 3.7e-16 / 9.3e-16 | 9.5e-16 / 7.7e-16 / 1.1e-15 | 4.4e-15 / 3.5e-15 / 4.8e-15 |
| | lj | 3.0e-16 | 4.3e-16 | 4.5e-16 | 4.4e-15 |
| D 10 MD steps, rebuild each step, original vs new | sw / tersoff | 1.2e-13 / 2.0e-13 | 1.2e-15 / 1.1e-15 | 1.0e-13 / n.c. | 3.4e-16 / 5.1e-16 |
| | eam / alloy / fs | 1.8e-13 / 1.1e-13 / 1.4e-13 | 2.1e-15 / 2.7e-15 / 2.1e-15 | 5.5e-14 / 5.9e-14 / 6.0e-14 | 1.4e-14 / 1.3e-14 / 4.9e-14 |
| | lj | 1.5e-13 | 2.7e-15 | 1.0e-13 | 1.3e-14 |
| C new path vs full-atom LAMMPS, per atom | sw / tersoff | 8.9e-13 / 9.4e-13 | 1.7e-14 / 1.6e-14 eV | | |
| | eam / alloy / fs | 5.9e-13 / 5.9e-13 / 5.6e-13 | 1.0e-14 / 8.9e-15 / 7.1e-15 eV | | |
| | lj | 6.4e-13 | 1.4e-14 eV | | |
Also: nodal forces (what the integrator uses) of original and new path are byte-identical in the 6-digit text dump at every step (`cmp`) for all styles/models at 1 rank; global virial pressure of the all-atom models equals full-atom LAMMPS to 2e-15..7e-15 (sw 1.5e-15, tersoff 6.3e-15, eam 3.2e-15, eam/alloy 2.1e-15, eam/fs 6.6e-15, lj 1.8e-15).
Why the results are not bit-identical: the original code sums, for each quadrature point, over a neighbour list ordered by the isoparametric scan; the new code sums over a cell-grid ordered list (EAM/LJ), or scatters center contributions into thread-private buffers (SW/Tersoff). Floating-point addition is not associative, so each force differs in the last 1-3 bits (a handful of ulp of the *terms*, up to ~1e-13 relative to the largest force when many terms of opposite sign cancel). For a fixed rank/thread count the result is deterministic (static scheduling), `new 1x1 vs 1x1` is bit-identical. Cross-checks against full-atom LAMMPS (an independent implementation) show the same level of agreement, i.e. the new path is as close to the exact result as the original.
Differences that are NOT round-off (original code behaviour, documented above): Tersoff per-point virial of the original code does not sum to the total virial (sum of xx = 60308 vs 489 for the correct one), so the per-point virial is not compared for Tersoff; the global virial pressure is compared with full-atom LAMMPS instead.
Element models (e, m, s3): the global virial pressure of CAC differs from full-atom LAMMPS by factors of ~3 for all styles (for eam, eam/alloy, eam/fs and lj the original and the new code give the same pressure to 1e-14, so it is not a regression; for sw/tersoff the original path does not tally the virial, so no comparison is possible) (nodal virial interpolation and its scaling by the number of cells per node, `(scale product)/nodes` with integer division, are unchanged); not a regression, but a property of the original method that may deserve a look.

### Verification results, NEP and the new GPU styles (same matrix: models a/e/m/s3, 1/2/4/8 MPI ranks, 1 and 4 OpenMP threads; GPU = NVIDIA L4, all ranks on the one GPU)
Counts: nep (CPU) 105/105, nep/kk (GPU, fp64) 105/105 (+61/61 in the final rerun with the virial convention note below), cac/sw/kk 61/61, cac/tersoff/kk 61/61, cac/eam/kk 61/61, cac/eam/alloy/kk 61/61, cac/eam/fs/kk 61/61, cac/lj/kk 61/61.
Worst-case relative deviations:
| check | style | force | energy | virial | thermo |
|---|---|---|---|---|---|
| A original vs new (same ranks) | nep CPU / nep/kk | 1.2e-13 / 1.5e-13 | 2.3e-15 / 2.5e-15 | 5.0e-14 / 1.0e-13 | 9.5e-14 / 2.2e-13 |
| B N ranks / T threads vs 1/1 | nep CPU / nep/kk | 9.4e-15 / 1.0e-14 | 4.5e-16 / 2.3e-16 | 2.5e-15 / 9.1e-15 | 1.1e-14 / 2.8e-14 |
| | sw/kk, tersoff/kk | 3.4e-14, 3.5e-14 | 1.0e-15, 5.4e-16 | 3.2e-14, 3.1e-14 | 7.0e-15, 1.1e-14 |
| | eam/kk, alloy/kk, fs/kk, lj/kk | <= 6.4e-16 | <= 1.4e-15 | <= 1.1e-15 | <= 5.9e-15 |
| D 10 MD steps, rebuild each step, original vs new | nep CPU / nep/kk | 2.6e-13 / 2.1e-13 | 2.4e-15 / 2.1e-15 | 6.4e-14 / 1.5e-13 | 1.4e-13 / 7.0e-13 |
| E GPU style vs CPU style of the same potential (new path) | sw/kk, tersoff/kk | 2.2e-13, 2.2e-13 | 3.1e-15, 2.9e-15 | 2.2e-13, 2.0e-13 | 1.7e-14, 2.8e-14 |
| | eam/kk, alloy/kk, fs/kk | 1.4e-13, 1.6e-13, 1.3e-13 | ~2e-15 | ~5e-14 | <= 3.9e-15 |
| | lj/kk, nep/kk | 1.5e-13, 1.0e-13 | 4.5e-15, 1.9e-15 | 1.0e-13, (see below) | 3.6e-15 |
| C vs full-atom LAMMPS, per atom | all 14 variants | <= 9.4e-13 (relative to mean |F|) | <= 2.2e-14 eV | | |
Notes: (1) The GPU SW/Tersoff kernels scatter with atomics, so their sums have no fixed order: results on the GPU are reproducible to ~1e-14, not bitwise. (2) NEP per-point virial: the CPU NEP and the Kokkos NEP engine use different per-atom virial partitions (CPU: all of a pair's virial on the neighbour, Kokkos: split between the two atoms); forces, energies and the global virial of atom models agree, the per-point virial and the element-model pressure do not, so those are excluded from the CPU-vs-GPU NEP comparison (they are compared separately against the original code of the same engine: A, 1.0e-13).

## Part D: host-side rebuild and per-step work (profiled on 1M atom-equivalent models, `models_classic/{cu,si,aln}_big_{e,m}.data`)
Profiling the 1M-atom-equivalent mixed Cu model (131072 atoms + 448 elements, 518k quadrature points, 1.05M sites) on one L4 + 16 CPU threads showed that, once the potential kernels are fast, CAC's own host code dominates both the step and the rebuild. Findings and fixes (each verified with the matrix of Part C afterwards; nothing changes the results beyond round-off, nodal forces stay byte-identical to the original code):
| item | before | after | change |
|---|---|---|---|
| per-step "Other" (AtomVecCAC::force_clear + check_distance_function: serial loops over every node, every step) | 15.4 ms/step | 1.5 ms/step | OpenMP; check_distance keeps an early-out flag |
| fix cac/nve initial/final integrate, nodal projection | serial | OpenMP (pinned threads: `OMP_PROC_BIND=true OMP_PLACES=cores`, without it the small loops were 4x slower) | |
| NPairCAC::build, element scan (only used for the interface flags in site-only mode) | 539 ms/rebuild | 8 ms (EAM) / 28 ms (LJ) | (1) atoms skip the scan; (2) stop at the first neighbour that is >= 2x finer; (3) skip bins that cannot hold one (per-bin minimum element volume + a separable min-filter over the stencil box); (4) cheap exact bounding-box prefilter before CAC_decide_element2element; (5) **the per-overlap-bin flag reset re-cleared all previously searched bins (quadratic); now only the new ones** (identical behaviour, also helps the original path) |
| PairCAC::build_site_table | 54 ms | 15 ms | site-only mode numbers the sites directly per element (`off[e] + ((p*n0+c0)*n1+c1)*n2+c2`), table and QP->site map filled in parallel; no hash/lookup |
| update_site_positions | 10 ms | 1.4 ms | OpenMP |
| device neighbour lists (GPU) | 42 ms | 18 ms | candidates screened in fp32 with a conservative margin and confirmed in fp64 (exact), count+fill passes merged by writing into a padded array sized from the previous rebuild (retry only if a list outgrows it); host grid sort partly threaded |
| static device upload after a rebuild | 87 ms | 1 ms | cached; the check compares owner/poly/type/quad maps only (weights follow from the element scales) |
| node packing, quadrature download | serial | OpenMP | |
New diagnostic switches: `CAC_TIMING=1` (phase times), `CAC_TIMING_DETAIL=1`, `CAC_TIMING_NPAIR=1` (NPairCAC stages), `CAC_DUMP_QUAD=<prefix>` (per-point dump from both the original and the new path).
The L4 has 1/64 fp64 throughput; the 1M-atom EAM device kernels (density + force, fp64) take 33 ms/step there, i.e. ~18 GFLOP at 0.47 TFLOPS, so on this card the step is kernel-bound; on A100/H100/B200 (fp64 at 1/2 rate) the same kernels would be ~10-30x faster and the host side (now 5-10 ms) would dominate.

## Part E: final state (2026-10-26)

* Final verification, full matrix (models a,e,m,s3,f,g; 1/2/4/8 ranks; 1/4 threads; checks A-E): CPU sw 144/144, tersoff 144/144, eam, lj, nep, eam/alloy, eam/fs 157/157 each; GPU 7 styles 91/91 each (1,710 comparisons). EAM family re-run on the final sorted-list binary (build_cuda2): 157/157 x3.
* Quick matrix of all 14 style variants on the final canonical build_cuda: all pass (CPU 49/49 with byte-identical nodal forces, GPU 29/29). Log: logs/final_quick.log.
* List sorting: default on for EAM only (27-35% faster steps, ~0.5 s extra per rebuild at 1M atoms); CAC_SORT_LISTS=0 disables. Device sort gave no GPU gain (CAC_SORT_LISTS_DEV opt-in).
* Final benchmark tables: logs/report_tables_final.md. Summary report for the meeting: MEETING_REPORT.md.
* Open: multi-rank/multi-GPU scaling not benchmarked; A100/H100 not measured; GPU SW/Tersoff not bitwise reproducible; flux/sector/ghost-quadrature need CAC_LEGACY_LISTS=1.
