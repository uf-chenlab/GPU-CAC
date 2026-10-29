#!/bin/bash
# incremental rebuild of build_cuda inside the interactive allocation:  srun --jobid=<id> --overlap bash tools/rebuild.sh [reconf]
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
export OMPI_CXX=$R/lammps_cac2026/lib/kokkos/bin/nvcc_wrapper NVCC_WRAPPER_DEFAULT_COMPILER=g++
cd $R/build_cuda
if [ "$1" = reconf ]; then cmake . -DBUILD_OMP=on -DKokkos_ENABLE_OPENMP=off 2>&1 | grep -iE "openmp|error" ; fi
make -j16 2>&1 | grep -E "error|Error|Linking|Built target lmp" | head -40
ls -la $R/build_cuda/lmp
echo BUILD_DONE
