# CAC + NEP + Kokkos: status (2026-10-26)

Local working copy: `~/projects/automate_uf3/cac_kokkos` (WSL). HPG copy: `/blue/ypchen/ntaormina/nep/cac_kokkos`.

## Layout
- `lammps_cac2026/`: LAMMPS 30 Mar 2026 (commit 457ea8c, the same base as our nep/kk tree) plus CAC, NEP and Kokkos changes. The source tree only; no `.git`, docs or examples.
- `cac_port.bundle`: git bundle of branch `cac-port` (8 commits on top of 457ea8c), with the full history and per-step messages.
  - Restore: `git clone https://github.com/lammps/lammps.git && cd lammps && git fetch ../cac_port.bundle cac-port:cac-port && git checkout cac-port`
- `bench/`: inputs, models, logs.
  - `run_suite.sh` + `compare_suite.py`: regression suite, original vs new.
  - `run_bench.sh`: benchmarks. `bench_orig/` = original 2021 CAC, `bench_n3/` = current build.
  - `nep/`: NEP validation inputs and logs.
- `tools/`:
  - `make_cac_model.py`: generates CAC data files (fcc, diamond, wurtzite; atomistic core plus element shell).
  - `expand_cac.py`: expands a CAC model to full atoms.
  - `compare_quad.py`: compares quadrature-point forces with full-atom NEP.
- `audit/core_diff.patch`: every core-LAMMPS change the original USER-CAC branch made.
- `test/` (HPG only): inputs for the GPU validation.

## Done
1. **Audit of the original CAC** (Adrian-Diaz/lammps USER-CAC, 27 May 2021, about 51k lines).
   - Mapped all core hooks.
   - Bugs found and fixed:
     - 32 missing-return undefined-behaviour functions;
     - an `asa_error` buffer overflow;
     - wrong `%ld` formats in the dumps;
     - the `proj_c2a` (b·a instead of c·a) neighbour-scan bug for non-orthogonal elements;
     - uninitialised `imagedata` in `data_CAC`;
     - the atom-style leaks;
     - `CommCAC` shadowing about 45 `CommTiled` members, so unsupported comm paths now error out instead of crashing.
   - Behaviours to note (not changed): `velocity/cac` only sets nodal velocities with `loop geom`; the virial scaling uses integer division; `ev_tally_full` was called with uninitialised arguments (zeros are now passed).
2. **Port to LAMMPS 2026.**
   - Core changes (about 500 lines, 23 files):
     - a comm-style registry (`style_comm.h`, `COMM_CLASS`);
     - the `CAC Elements` read_data section;
     - `NB/NS/NP_CAC` and `REQ_CAC` neighbour flags;
     - the atom-vec rebuild check and stencil post-create hooks;
     - min copy hooks;
     - the `set cac/*` keywords.
   - The package now lives in `src/CAC`.
   - **The port is bit-identical to the original on 10 regression cases**: eam, eam/alloy, eam/fs, lj, tersoff, sw, 4 MPI ranks, CG minimisation and dumps.
3. **CPU optimisation**, all verified against the original to 1e-16 relative (round-off):
   - A site table: unique lattice sites are deduplicated, interpolated once per step, and the per-quadrature-point lists become gathers.
   - A memoised EAM site kernel: density and F′ computed once per site, instead of O(N_in·(N_in+N_out)) work per quadrature point.
   - An LJ site kernel.
   - A constant Q8 inverse mass matrix replaces the per-element mass matrix plus LU solve every step.
   - An exact nodal rebuild check replaces an `asa_cg` optimisation per element per step.
   - In the neighbour build, positions along a row are linear in s, and the scan uses exact sphere intervals.
   - A dense site lookup, and site lists reuse the inner lists.
   - Speedups (Loop time, 1 core):

     | Case | Original (s) | New (s) | Speedup |
     |---|---|---|---|
     | Cu mid EAM | 72.1 | 4.07 | 17.7× |
     | Cu small EAM | 61.3 | 3.39 | 18× |
     | Cu block example, loop | 325 | 32.6 | 10× |
     | Cu block example, wall | 7:29 | 1:10 | |
     | LJ | 21.4 | 4.3 | 5× |
     | 4-rank EAM | 28.3 | 1.46 | 19× |
     | Tersoff | 11.4 | 9.5 | 1.2× |
     | SW | 9.4 | 7.4 | 1.3× |

4. **NEP.**
   - The NEP package from the HPG tree is now a proper package: `src/NEP`, with `pair_NEP` renamed to `pair_nep` and the debug `printf` of every force removed.
   - New `pair_style cac/nep [one]`, CPU, via `NEP3::compute_for_lammps` on the CAC sites.
     - Quadrature-point forces match full-atomistic `pair_style nep` to 5e-13 eV/Å.
     - Tested on a mixed atom/element AlN model with ZBL, serial and 4 MPI ranks.
5. **Kokkos.**
   - The nep/kk kernels were factored into `KOKKOS/nep_kokkos_engine.h`. nep/kk stays bitwise identical: checked against a reference build.
   - New `pair_style cac/nep/kk [one] [fp64|fp32|df64]`.
     - CAC stays on the host. Each step it uploads nodal positions, interpolates sites on the device, runs the NEP kernels on the site set, and gathers the quadrature forces back.
     - Run it without `-k on`; it initialises Kokkos itself.
     - Validated with the Kokkos **OpenMP** backend against full-atom NEP: fp64 4.6e-13, df64 3.3e-12, fp32 8e-6 eV/Å.
     - Speed: 8 threads run 9.9× faster than CPU cac/nep on 1 rank, and 3.4× faster than 8 MPI ranks.

6. **GPU (CUDA) build and validation of cac/nep/kk** (2026-10-26).
   - Build: `tools/build_cuda.sbatch` (job on hpg-turin, phillpot QOS, 16 cores, about 14 min) -> `build_cuda/lmp`. Kokkos CUDA, `Kokkos_ARCH_ADA89`, C++20, CUDA 12.8.1.
   - **Environment matters.** Use the env of `proj/new_workflow/gpuq/env.sh`: `module load gcc openmpi/5.0.10 cuda/12.8.1`, then put the non-CUDA `/apps/mpi/gcc/14.2.0/openmpi/5.0.10` bin/lib on PATH/LD_LIBRARY_PATH, and `OMPI_MCA_pml=ucx UCX_TLS=^ib`. The CUDA variant `openmpi/5.0.7` (used by the old validation scripts and by `~/bin/lmp_kokkos_nep`) needs `libhcoll.so.1`, which is missing on the nodes tested, so it neither links nor starts. `mpicxx` from the CUDA openmpi also crashes (illegal instruction) on hpg-default AMD Rome nodes.
   - Validation (`tools/validate_gpu.sbatch`, 1 L4, `test/gpu_run/`): 3336 quadrature points vs full-atom NEP. Max force error: fp64 4.5e-13, df64 3.3e-12, fp32 1.0e-5 eV/A. Max site-energy error: 2.3e-14, 1.6e-13, 1.4e-6 eV.
   - Benchmark (`tools/bench_gpu.sbatch`, `test/gpu_bench/`, 8272-atom-equivalent AlN, 20 steps, 1 L4, 1 rank): Loop time fp64 0.404 s, df64 0.257 s, fp32 0.132 s. CPU cac/nep, 1 rank: 56.6 s (140x, 220x, 430x). Kokkos OpenMP, 8 threads: 5.34 s (13x, 21x, 40x). GPU utilisation peaked at 90%. The case is small, so the GPU is not saturated; larger models are untested.

## Remaining
1. **Markdown report** (audit, port, optimisations, benchmarks) for the user/group. The numbers are in `bench/` and above.
2. **Optional next optimisations.**
   - Test cac/nep/kk on a large model (360k-atom equivalent) and on multiple GPUs / MPI ranks.
   - Tersoff/SW site kernels: they still use the per-quadrature-point path, so most of their time is gathers.
   - OpenMP threading of the EAM/LJ site kernels.
   - Building site ids directly in NPairCAC to cut the memory of the quadrature lists. It is about 12 GB for 360k-atom-equivalent NEP models: 384M list entries.
3. **Physics notes to follow up.**
   - With NEP's 8 Å cutoff, the CAC surface-quadrature layers (2rc+skin ≈ 18 Å) cover most of an element unless elements are much larger than about 20 Å/side. Coarse-graining gains need large elements.
   - In NVE the nodal temperature drifts (as in the original EAM/SW runs). An all-atom CAC NEP vs LAMMPS NEP trajectory comparison has not been run yet.

## Update 2026-10-26
CAC audit/optimisation finished: site-only lists, threaded CPU kernels, GPU classical styles (cac/{lj,eam,eam/alloy,eam/fs,sw,tersoff}/kk), faster cac/nep and cac/nep/kk. All 14 style variants verified against the original code and full-atom LAMMPS. See MEETING_REPORT.md (summary) and OPTIMIZATION_LOG.md (details).
