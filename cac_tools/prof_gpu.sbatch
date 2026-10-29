#!/bin/bash
#SBATCH --job-name=cac_prof
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16gb --time=00:40:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/prof_%j.log
# Same env as proj/new_workflow/gpuq/env.sh (openmpi/5.0.10 non-CUDA build)
module purge; module load gcc; module load openmpi/5.0.10; module load cuda/12.8.1; module load conda
source activate neptools_env
export LD_PRELOAD=/blue/ypchen/ntaormina/bin/miniconda38/envs/neptools_env/lib/libstdc++.so.6
export PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_TLS=^ib
R=/blue/ypchen/ntaormina/nep/cac_kokkos
LMP=$R/build_cuda/lmp
nvidia-smi -L
cd $R/prof
export OMPI_MCA_btl_vader_single_copy_mechanism=none
which perf
echo "=== setup-only (run 0)"
perf record -F 200 -g -o perf_setup.data mpirun -np 1 $LMP -in in.setup -var ps "cac/nep/kk one fp64" -log none > setup.out 2>&1
grep "Total wall" setup.out
perf report -i perf_setup.data --no-children --sort symbol -g none --stdio 2>/dev/null | grep -v "^#" | grep -v "^$" | head -40
echo "=== with 20 steps"
perf record -F 200 -g -o perf_step.data mpirun -np 1 $LMP -in in.step -var ps "cac/nep/kk one fp64" -log none > step.out 2>&1
grep -E "Total wall|Loop time" step.out
perf report -i perf_step.data --no-children --sort symbol -g none --stdio 2>/dev/null | grep -v "^#" | grep -v "^$" | head -30
