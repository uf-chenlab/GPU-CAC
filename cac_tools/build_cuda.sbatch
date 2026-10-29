#!/bin/bash
#SBATCH --job-name=cac_rebuild
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/build_cuda_%j.log
set -e
module purge
module load gcc
module load openmpi/5.0.10
module load cuda/12.8.1
module load cmake
# env as in proj/new_workflow/gpuq/env.sh: the non-CUDA openmpi/5.0.10 (libhcoll-free), put on PATH by hand
export PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/lib:$LD_LIBRARY_PATH
module list 2>&1
which mpicxx nvcc
cd /blue/ypchen/ntaormina/nep/cac_kokkos/build_cuda
export OMPI_CXX=/blue/ypchen/ntaormina/nep/cac_kokkos/lammps_cac2026/lib/kokkos/bin/nvcc_wrapper
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake ../lammps_cac2026/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin/mpicxx -DCMAKE_CXX_STANDARD=20   -DPKG_CAC=on -DPKG_NEP=on -DPKG_MANYBODY=on -DPKG_KOKKOS=on -DKokkos_ENABLE_CUDA=on -DKokkos_ARCH_ADA89=on -DKokkos_ENABLE_OPENMP=off -DBUILD_OMP=off
make -j16
ls -la lmp
