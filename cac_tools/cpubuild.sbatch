#!/bin/bash
#SBATCH --job-name=cac_cpubuild
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/cpubuild_%j.log
set -e
module purge
module load gcc openmpi/5.0.10 cmake
export PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_TLS=^ib
W=/blue/ypchen/ntaormina/nep/cac_kokkos/anvil_test
mkdir -p $W && cd $W
tar xzf ../anvil_pkg/cac_src.tar.gz
tar xzf ../anvil_pkg/bench_kit.tar.gz
mkdir -p build_cpu && cd build_cpu
cmake ../lammps_cac2026/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DPKG_CAC=on -DPKG_NEP=on -DPKG_MANYBODY=on -DBUILD_MPI=on -DBUILD_OMP=on \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" 2>&1 | tail -5
make -j16 2>&1 | tail -5
ls -la lmp
cd ../kit
PY=/blue/ypchen/ntaormina/bin/miniconda38/envs/neptools_env/bin/python
CU="--lattice fcc --a 3.615 --masses 63.546"
SI="--lattice diamond --a 5.431 --masses 28.0855"
$PY make_cac_model.py $CU --blocks 4 4 4 --block 4 --core 0.0  -o cu_small_e.data | tail -1
$PY make_cac_model.py $CU --blocks 4 4 4 --block 4 --core 0.25 -o cu_small_m.data | tail -1
$PY make_cac_model.py $SI --blocks 4 4 4 --block 4 --core 0.0  -o si_small_e.data | tail -1
$PY make_cac_model.py $SI --blocks 4 4 4 --block 4 --core 0.25 -o si_small_m.data | tail -1
export LMP=$W/build_cpu/lmp
for c in new old; do
  CODE=$c NP=4 THREADS=1 MODELS="cu_small_e cu_small_m si_small_e si_small_m" NS_STATIC=5 bash anvil_bench.sh
done
cat results.tsv
