#!/bin/bash
#SBATCH --job-name=cac_gpu_val
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16gb --time=00:30:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/gpu_val_%j.log
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
cd $R/test/gpu_run
for prec in fp64 df64 fp32; do
  echo "=== cac/nep/kk one $prec"
  rm -f q_$prec.*
  CAC_DUMP_QUAD=q_$prec mpirun -np 1 $LMP -in in.cacnep_v -var ps "cac/nep/kk one $prec" -log log_$prec.lammps 2>&1 | tail -25
  python $R/tools/compare_quad.py q_$prec $R/test/full_forces.dump
done
