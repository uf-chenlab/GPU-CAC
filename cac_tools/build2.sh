#!/bin/bash
# second build tree (for experiments while build_cuda is in use by running tests):  srun --jobid=<id> --overlap bash tools/build2.sh
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
export OMPI_CXX=$R/lammps_cac2026/lib/kokkos/bin/nvcc_wrapper NVCC_WRAPPER_DEFAULT_COMPILER=g++
cd $R/build_cuda2
if [ ! -f CMakeCache.txt ]; then
  cmake ../lammps_cac2026/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin/mpicxx -DCMAKE_CXX_STANDARD=20 \
    -DPKG_CAC=on -DPKG_NEP=on -DPKG_MANYBODY=on -DPKG_KOKKOS=on -DKokkos_ENABLE_CUDA=on -DKokkos_ARCH_ADA89=on -DKokkos_ENABLE_OPENMP=off -DBUILD_OMP=on > cmake.log 2>&1
fi
make -j16 2>&1 | grep -E "error|Error|Built target lmp" -A4 | head -30
ls -la $R/build_cuda2/lmp
echo BUILD2_DONE
