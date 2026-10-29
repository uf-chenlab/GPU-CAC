#!/bin/bash
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
export OMPI_CXX=$R/lammps_cac2026/lib/kokkos/bin/nvcc_wrapper NVCC_WRAPPER_DEFAULT_COMPILER=g++
cd $R/build_cuda
cmake . > /dev/null 2>&1
make -j16 2>&1 | grep -E "error|Error|undefined|Linking|Built target lmp" -A4 | head -80
ls -la $R/build_cuda/lmp
echo BUILD_DONE
