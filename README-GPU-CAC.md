# GPU-CAC

LAMMPS (30 Mar 2026 base) with the CAC (Concurrent Atomistic-Continuum) package, the NEP potential, and Kokkos (CUDA) versions of the CAC pair styles.
This tree is a complete LAMMPS source tree; build it with cmake as usual.

Start here:
* `docs/MEETING_REPORT.md`: summary of the CAC efficiency audit, verification and benchmarks (CPU and GPU).
* `docs/OPTIMIZATION_LOG.md`: chronological log with every measurement and the bugs found in the original CAC code.
* `docs/STATUS.md`: state of the CAC + NEP + Kokkos work and how to build.
* `cac_tools/`: benchmark, verification and model-generation scripts (`make_cac_model.py`, `bench_classic.sh`, `verify_all.sh`, ...), test inputs (`test_classic/`), small test models and potentials (`models/`), and a portable CPU benchmark script (`anvil/anvil_bench.sh`). Paths inside the scripts are those of the HiPerGator work area and need adjusting.

Packages: `src/CAC`, `src/NEP`, `src/KOKKOS/pair_cac_*_kokkos.*`, `src/KOKKOS/cac_site_device.h`.
CUDA build flags used (L4, Ada89): `-DPKG_CAC=on -DPKG_NEP=on -DPKG_MANYBODY=on -DPKG_KOKKOS=on -DKokkos_ENABLE_CUDA=on -DKokkos_ARCH_ADA89=on -DKokkos_ENABLE_OPENMP=off -DBUILD_OMP=on -DCMAKE_CXX_STANDARD=20`.
CPU-only build: same without the Kokkos options.
Original code path at run time: `CAC_LEGACY_LISTS=1`.

State: snapshot of 2026-10-26. The original USER-CAC code is by Adrian Diaz (https://github.com/Adrian-Diaz/lammps/tree/USER-CAC).
