# CAC efficiency audit and optimisation: report for the group meeting

Work area: `/blue/ypchen/ntaormina/nep/cac_kokkos` on HiPerGator (LAMMPS 30 Mar 2026 base + CAC package + NEP + Kokkos).
Chronological details, every intermediate measurement and every bug found are in `OPTIMIZATION_LOG.md` (same folder); this file is the summary.
All measurements: NVIDIA L4 GPU (hpg-turin), 16 CPU cores, one MPI rank unless stated. The L4 has 1/64 fp64 throughput, so GPU numbers are a lower bound for A100/H100/B200 class cards.

## 1. Summary

* **Where the time went.** For every potential the *MD step* was already cheap compared with the cost of *rebuilding the CAC neighbour lists* (0.3 to 2.4 s per rebuild for 16k atoms, i.e. 30 to 150 MD steps; 9 to 13 s for the NEP benchmark), and the CPU NEP kernel was single threaded and recomputed every radial function several times. Per-point CAC kernels for Stillinger-Weber and Tersoff recomputed every three-body term for each of its three participants.
* **What was done.**
  1. *Site-only neighbour lists.* Every quadrature point is a lattice site, so the quadrature neighbour lists do not need the expensive per-point scan through isoparametric space with ASA minimisation. All lattice sites of local and ghost elements are numbered directly and the lists come from a sorted cell grid. This removed the dominant cost for all potentials.
  2. *Site kernels.* SW and Tersoff now compute each term once per centre and scatter the forces (threaded on the CPU, atomics on the GPU); EAM and LJ read the new lists.
  3. *Threading.* OpenMP for the NEP CPU kernel (fused, one pass per neighbour instead of three), the site kernels, nodal projection, integrator, force clearing and the displacement check (the last two were 15 ms of serial work per step for 1M atoms).
  4. *GPU versions of all classical styles* (`cac/lj/kk`, `cac/eam/kk`, `cac/eam/alloy/kk`, `cac/eam/fs/kk`, `cac/sw/kk`, `cac/tersoff/kk`) on top of one shared device component (node upload, site interpolation, neighbour lists built on the device with an fp32-screened/fp64-confirmed cell search); `cac/nep/kk` uses the same ideas. They run without `-k on`.
  5. *Rebuild-path fixes* in the LAMMPS-side code (`NPairCAC::build`): the interface-flag scan (539 ms to 8 ms on 1M atoms), a quadratic flag reset, the site-table construction (parallel, direct numbering), cached device uploads.
* **Headline numbers** (start of the work vs now, 16k atom-equivalents; full tables in section 4): MD step up to **82x faster on 16 CPU threads** (SW, 45 to 0.56 ms) and **60x on the GPU**; neighbour rebuild **20 to 300x faster on the CPU and 20 to 1700x on the GPU** (EAM/fs on elements: 2.4 s to 1.4 ms). NEP: CPU step 2.83 s (1 thread) to 0.11 s (16 threads); GPU rebuild 13 s to 10-19 ms; GPU fp32 step 3.6 to 5.3 ms.
* **Correctness.** The new code reproduces the original USER-CAC results (and full-atom LAMMPS) to floating-point round-off: per-quadrature-point force/energy/virial deviations of at most 3e-13 relative (typically 1e-15), nodal forces byte-identical to the original at every step, for 14 style variants (6 classical CPU, 6 classical GPU, NEP CPU/GPU), atom/element/mixed models (including elements of different sizes and odd sizes), 1 to 8 MPI ranks, 1 to 4 threads (section 5). The checks also found several bugs in the original code (section 6).

## 2. Findings from the profiling

| observation | consequence |
|---|---|
| one neighbour rebuild cost 0.3 to 2.4 s (16k atoms), 9 to 13 s (NEP benchmark), 85% of it in `NPairCAC::neighbor_accumulate` + ASA solves + shape functions | replaced by direct site numbering and cell-grid lists (section 3.1) |
| CPU NEP: 3 loops over all 340 neighbours per centre, radial basis recomputed for each of the 5 angular orders, single threaded | fused single pass, stored derivatives, OpenMP |
| SW/Tersoff: every triplet term evaluated once for each of its 3 atoms, per-point cluster lists rebuilt every step | centre kernels with scatter |
| per step, 15 ms of serial CAC host loops at 1M atoms (force clearing, `check_distance_function`) in *every* style | threaded |
| GPU step for classical potentials on the L4: 33 of 46 ms are device kernels (fp64 at 1/64 rate) | hardware bound here; host side now 5 to 10 ms |
| the OpenMP default thread placement made the small host loops 4x slower | run with `OMP_PROC_BIND=true OMP_PLACES=cores` and `mpirun --bind-to none` |

## 3. What changed (files under `lammps_cac2026/src`)

### 3.1 Core (CAC package)
* `CAC/npair_cac.cpp`, `atom.h/.cpp` (`inner_only_flag`): *site-only mode*. 1 = lists with halved cutoff; 2 = no per-point lists at all (many-body styles: ghost region 2 rc + skin); 3 = same for single-range styles (LJ). `NPairCAC::build` then only decides the interface flags (which elements need full sampling next to atoms or finer elements) with an exact, early-exit, bin-skipping scan. The sampled surface layer keeps its original depth (2 rc for many-body styles) so the quadrature point set and weights are identical to the original.
* `CAC/pair_cac_sites.cpp`, `pair_cac.h/.cpp`: `build_site_table_fast()` (sites numbered per element, threaded), `build_site_grid()` (sorted cell grid), `build_site_lists()` (threaded count-then-fill into the final CSR array; optional sort by site id, default on for EAM where it gives 27 to 35% faster steps), scatter helpers `scatter_begin/finish`, per-point dump `CAC_DUMP_QUAD`.
* `CAC/pair_cac_{eam,lj}.cpp`: kernels read `quad_list()`; threaded; `pair_cac_sw.cpp`, `pair_cac_tersoff.cpp`: new `compute_sites()` (centre + scatter, same formulas and same energy/virial partition as the original code).
* `CAC/atom_vec_cac.cpp`, `fix_cac_nve.cpp`: OpenMP.
* `NEP/nep.cpp`: `compute_for_lammps_fast` (fused, threaded; `NEP_CPU_LEGACY=1` selects the old routines).
* `CAC_LEGACY_LISTS=1` restores the original per-point scan and kernels at run time (used for all A/B checks).

### 3.2 Kokkos (GPU) package
* `KOKKOS/cac_site_device.h`: shared device component (static site table cached between rebuilds, node upload, site interpolation, cell-grid neighbour lists on the device, result download, scatter accumulators).
* `KOKKOS/pair_cac_{lj,eam,sw,tersoff}_kokkos.{h,cpp}`: the six new GPU styles; `pair_cac_nep_kokkos.*` uses the same list machinery.

### 3.3 Tools (`tools/`)
`bench_classic.sh`, `permsteps.py`, `report_tables.py` (benchmarks); `verify_all.sh`, `verify_pq.py`, `validate_classic.sh`, `force_check.sh`, `summarize_verify.py` (verification); `make_cac_model.py`/`expand_cac.py`/`gen_models.sh`/`gen_big.sh` (models, now including mixed element sizes); `build_cuda.sbatch`, `rebuild.sh` (build).

## 4. Performance

`step` = ms per MD step, `rebuild` = ms for one neighbour-list rebuild. base = start of this work (original CAC code with the earlier single-thread optimisations), cpu1/cpu16 = new code with 1/16 OpenMP threads, gpu = L4 + 16 CPU threads. Benchmarks `tools/bench_classic.sh`: 20 MD steps, and 5 steps with a rebuild at every step; models: 16384 (Cu) / 13824 (Si) atom-equivalents as all atoms, all elements (4x4x4 cells), or atoms+elements.


### Classical potentials, 13.8k-16.4k atom-equivalents, one rank

`step` = ms per MD step, `rebuild` = ms per neighbour-list rebuild (everything that happens when LAMMPS rebuilds the lists). base = start of this work, cpu1/cpu16 = new code with 1 / 16 OpenMP threads, gpu = NVIDIA L4 (fp64) + 16 CPU threads.

| potential | model | step base | step cpu1 | step cpu16 | step gpu | rebuild base | rebuild cpu1 | rebuild cpu16 | rebuild gpu |
|---|---|---|---|---|---|---|---|---|---|
| EAM (Cu_u3) | all atoms | 30.8 | 28.1 | 2.52 | 1.57 | 334 | 42.9 | 7.1 | 3.0 |
| EAM (Cu_u3) | all elements | 16.2 | 18.3 | 1.66 | 1.25 | 1234 | 46.3 | 7.2 | 0.9 |
| EAM (Cu_u3) | atoms + elements | 22.2 | 20.8 | 1.99 | 1.27 | 1140 | 50.1 | 7.2 | 1.3 |
| EAM/alloy | all atoms | 25.7 | 18.9 | 1.65 | 1.67 | 392 | 70.4 | 9.7 | 3.1 |
| EAM/alloy | all elements | 22.3 | 18.2 | 2.10 | 1.45 | 1621 | 81.3 | 8.6 | 1.1 |
| EAM/alloy | atoms + elements | 24.9 | 23.1 | 2.22 | 1.49 | 1351 | 80.2 | 8.6 | 1.5 |
| EAM/fs | all atoms | 78.5 | 71.4 | 6.19 | 3.32 | 755 | 169.6 | 16.4 | 3.5 |
| EAM/fs | all elements | 55.9 | 54.6 | 5.22 | 3.08 | 2406 | 180.3 | 19.9 | 1.4 |
| EAM/fs | atoms + elements | 63.8 | 65.5 | 5.68 | 3.08 | 2075 | 180.3 | 18.8 | 1.8 |
| LJ 5.0 | all atoms | 4.7 | 5.3 | 0.62 | 0.72 | 70 | 14.9 | 3.8 | 3.1 |
| LJ 5.0 | all elements | 4.1 | 3.4 | 0.39 | 0.49 | 399 | 13.4 | 1.7 | 0.9 |
| LJ 5.0 | atoms + elements | 5.5 | 3.3 | 0.45 | 0.54 | 398 | 13.3 | 2.3 | 1.6 |
| Stillinger-Weber | all atoms | 29.5 | 4.8 | 0.70 | 1.03 | 101 | 13.2 | 3.3 | 2.5 |
| Stillinger-Weber | all elements | 45.1 | 3.6 | 0.56 | 0.79 | 573 | 10.3 | 2.1 | 0.7 |
| Stillinger-Weber | atoms + elements | 48.1 | 3.6 | 0.59 | 0.80 | 571 | 10.6 | 2.3 | 1.0 |
| Tersoff | all atoms | 88.0 | 17.7 | 1.54 | 3.52 | 68 | 5.4 | 2.7 | 2.4 |
| Tersoff | all elements | 100.1 | 17.6 | 1.48 | 2.32 | 463 | 3.6 | 1.5 | 0.7 |
| Tersoff | atoms + elements | 99.3 | 17.6 | 1.50 | 2.34 | 502 | 4.8 | 1.9 | 1.0 |

### Speed-up factors (base / new)

| potential | model | step: cpu1 | step: cpu16 | step: gpu | rebuild: cpu1 | rebuild: cpu16 | rebuild: gpu |
|---|---|---|---|---|---|---|---|
| EAM (Cu_u3) | all atoms | 1.1x | 12.2x | 19.6x | 7.8x | 46.9x | 111.6x |
| EAM (Cu_u3) | all elements | 0.9x | 9.7x | 13.0x | 26.6x | 171.5x | 1323.5x |
| EAM (Cu_u3) | atoms + elements | 1.1x | 11.2x | 17.4x | 22.8x | 157.8x | 885.9x |
| EAM/alloy | all atoms | 1.4x | 15.5x | 15.3x | 5.6x | 40.4x | 125.5x |
| EAM/alloy | all elements | 1.2x | 10.6x | 15.4x | 19.9x | 188.5x | 1438.0x |
| EAM/alloy | atoms + elements | 1.1x | 11.2x | 16.8x | 16.8x | 157.3x | 891.0x |
| EAM/fs | all atoms | 1.1x | 12.7x | 23.6x | 4.4x | 45.9x | 213.2x |
| EAM/fs | all elements | 1.0x | 10.7x | 18.2x | 13.3x | 120.8x | 1691.4x |
| EAM/fs | atoms + elements | 1.0x | 11.2x | 20.7x | 11.5x | 110.4x | 1149.4x |
| LJ 5.0 | all atoms | 0.9x | 7.5x | 6.5x | 4.7x | 18.4x | 22.5x |
| LJ 5.0 | all elements | 1.2x | 10.6x | 8.5x | 29.9x | 232.0x | 421.2x |
| LJ 5.0 | atoms + elements | 1.7x | 12.3x | 10.3x | 29.8x | 175.2x | 249.1x |
| Stillinger-Weber | all atoms | 6.1x | 41.9x | 28.7x | 7.6x | 30.8x | 39.9x |
| Stillinger-Weber | all elements | 12.7x | 80.6x | 57.1x | 55.8x | 277.7x | 778.3x |
| Stillinger-Weber | atoms + elements | 13.5x | 81.9x | 60.4x | 54.0x | 250.8x | 582.0x |
| Tersoff | all atoms | 5.0x | 57.0x | 25.0x | 12.6x | 24.9x | 28.7x |
| Tersoff | all elements | 5.7x | 67.6x | 43.1x | 128.6x | 304.6x | 651.2x |
| Tersoff | atoms + elements | 5.6x | 66.0x | 42.4x | 103.7x | 261.3x | 504.7x |

### One million atom-equivalents (1,048,576 Cu / 884,736 Si), CPU 16 threads vs L4 GPU

`all elements` = 512 (Cu) / 216 (Si) elements of 8x8x8 cells; `mixed` = 131,072 (Cu) / 32,768 (Si) atoms in the centre + elements.

| potential | model | step cpu16 (ms) | step gpu (ms) | rebuild cpu16 (ms) | rebuild gpu (ms) |
|---|---|---|---|---|---|
| EAM (Cu_u3) | all elements | 32.3 | 18.8 | 203 | 42 |
| EAM (Cu_u3) | mixed | 68.5 | 43.3 | 334 | 79 |
| EAM/alloy | all elements | 36.2 | 21.5 | 261 | 44 |
| EAM/alloy | mixed | 78.1 | 47.7 | 397 | 78 |
| EAM/fs | all elements | 81.9 | 52.3 | 924 | 86 |
| EAM/fs | mixed | 191.1 | 98.7 | 1154 | 112 |
| LJ 5.0 | all elements | 5.4 | 3.4 | 26 | 22 |
| LJ 5.0 | mixed | 19.0 | 15.3 | 114 | 94 |
| Stillinger-Weber | all elements | 23.7 | 9.9 | 59 | 20 |
| Stillinger-Weber | mixed | 27.2 | 13.0 | 101 | 53 |
| Tersoff | all elements | 65.1 | 39.9 | 37 | 19 |
| Tersoff | mixed | 71.3 | 43.9 | 89 | 65 |

### 4.4 NEP (`cac/nep`, `cac/nep/kk`), 49,152 atom-equivalents (8,272 atoms+elements), one rank

| | start of work | now |
|---|---|---|
| `cac/nep` CPU, 1 thread, ms/step | 2830 | 1490 (fused kernel) |
| `cac/nep` CPU, 16 threads, ms/step | n/a (single threaded) | 104-114 |
| `cac/nep` CPU rebuild, ms | 8600 + 740 | 25-84 |
| `cac/nep/kk` GPU fp64, ms/step | 20 | 18-20 (the NEP engine; identical to full-atom `nep/kk`) |
| `cac/nep/kk` GPU fp32, ms/step | 5 | 3.6-5.3 |
| `cac/nep/kk` GPU rebuild, ms | 9400-13000 | 10-19 |
| `cac/nep/kk`, 393,216 atom-equivalents: fp64 / fp32 ms/step, rebuild ms | n/a | 150 / 38-40, 80 |

Per step the GPU is within 1-2 ms of plain atomistic `nep/kk` on the same atoms; the remaining time is the NEP engine. CAC only pays off for NEP when the elements are much larger than twice the sampled surface layer (about 24 A for NEP rc = 8 A): in all models here every site is within a cutoff of a quadrature point, so the work equals the atomistic work.

## 5. Verification (details: `OPTIMIZATION_LOG.md`, part C; tools `tools/verify_all.sh`, `verify_pq.py`)

Models (jittered by 0.04 A so that forces are non-trivial): `a` all atoms, `e` all elements, `m` atoms+elements, `s3` odd element size (3x3x3 cells), `f` two element sizes (4 and 2 cells; tests the element/element interface logic), `g` atoms + two element sizes; Cu fcc, Si diamond, AlN wurtzite.
For every style x model x (1, 2, 4, 8 MPI ranks) x (1, 4 OpenMP threads) the per-quadrature-point force density, energy and virial of the new code are compared with
* **A** the original code path (`CAC_LEGACY_LISTS=1`, same ranks), point by point, plus nodal forces (text dump, 6 digits, `cmp` identical at every step) and thermo output;
* **B** the new code on 1 rank / 1 thread (matched by position);
* **C** full-atom LAMMPS on the equivalent atomistic configuration (forces and energy of every atom; global virial pressure);
* **D** 10 MD steps with a neighbour rebuild at every step, original vs new;
* **E** each GPU style against its CPU counterpart.
Result: **1,710 comparisons in the full matrix plus the final-binary re-runs, all passed**. Worst-case relative deviations: forces 3e-13 (typically 1e-15), energies 3e-15, virials 3e-13; vs full-atom LAMMPS forces 9e-13 and per-atom energy 2e-14 eV; global virial pressure of all-atom models equals LAMMPS to 2e-15..7e-15.
"Identical" here means identical up to floating-point round-off: the new kernels add the same terms in a different order (cell-grid neighbour order, scatter instead of gather, threads, atomics on the GPU). For a fixed number of ranks and threads the CPU results are deterministic (static scheduling; a 1x1 run is bit-identical to a repeated 1x1 run). The GPU SW/Tersoff kernels use atomics and are reproducible to ~1e-14, not bitwise.

## 6. Bugs found in the original CAC code (all reproduced and documented in the log)

1. `cac/sw` and `cac/tersoff` (original per-point path): the **total energy is wrong for models that mix atoms with elements of odd size** (-21,423 eV instead of -56,280 eV for the 13,824-atom test; full-atom LAMMPS agrees with the new value to 1e-14).
2. `cac/sw` and `cac/tersoff` (original path) do not tally the virial for atom-only models (pressure 0); the original Tersoff per-point virial does not even sum to the total (sum of xx = 60,308 vs 489). The new kernels reproduce the full-atom virial pressure.
3. The original non-site path **segfaults** in `PairCAC::LUPSolve` (virial projection) on MPI ranks whose first local element is an atom when the virial is requested (uninitialised LU pivots).
4. Information, not a regression: the global virial pressure of element models differs from full-atom LAMMPS by factors of about 3 in the original and the new code alike (nodal virial interpolation and its scaling by cells per node).
5. The NEP CPU and Kokkos engines use different per-atom virial partitions (global virial of atom models agrees).

## 7. How to run

```
module load gcc openmpi/5.0.10 cuda/12.8.1        # see tools/dev_env.sh; the non-CUDA openmpi build is put on PATH by hand
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMP_PLACES=cores
mpirun --bind-to none -np 1 build_cuda/lmp -in in.file
```
Binary: `build_cuda/lmp` (Kokkos CUDA, Ada89, `PKG_CAC PKG_NEP PKG_MANYBODY PKG_KOKKOS`, `BUILD_OMP=on`; build with `tools/build_cuda.sbatch`, incremental `tools/rebuild.sh`; second tree `build_cuda2` via `tools/build2.sh`).
GPU styles run without `-k on` (Kokkos is initialised by the style): `pair_style cac/eam/kk one`, `cac/eam/alloy/kk`, `cac/eam/fs/kk`, `cac/lj/kk 5.0 one`, `cac/sw/kk one`, `cac/tersoff/kk one`, `cac/nep/kk one [fp64|df64|fp32]`; same `pair_coeff` as the CPU styles. Several ranks can share one GPU.
Environment switches: `CAC_LEGACY_LISTS=1` original path; `CAC_SORT_LISTS=0|1` (list sorting, default on for EAM; turn off if you rebuild more often than every ~10 steps, the sort costs ~0.5 s per rebuild at 1M atoms); `NEP_CPU_LEGACY=1`; `CAC_TIMING=1` (+`CAC_TIMING_DETAIL=1`, `CAC_TIMING_NPAIR=1`) phase timers; `CAC_DUMP_QUAD=<prefix>` per-point dump (works in both paths).
Flux, sector and ghost-quadrature options still need `CAC_LEGACY_LISTS=1` (the style says so).

## 8. Limits and open items

* Not measured: MPI/multi-GPU **scaling** (correctness is verified for 1-8 ranks on one GPU, speed only for one rank), A100/H100/B200 (fp64-bound kernels will be 10-30x faster there; then the host side, now 5-10 ms per 1M atoms, dominates: moving the nodal projection and integrator to the device is the next step).
* The EAM list sort costs ~0.5 s per rebuild at 1M atoms; gains are 27-35% per step (EAM/alloy, EAM/fs).
* GPU SW/Tersoff are not bitwise reproducible (atomics).
* Elements pay off for NEP/EAM only if they are larger than about 2 cutoffs plus skin; the benchmarks use 16-29 A elements, so the CAC speed-up here comes from the faster algorithms, not from coarse-graining.
* Not addressed: buck, coul/dsf, coul/wolf, pb/dsf styles still use the original per-point path.
