#!/bin/bash
#SBATCH --job-name=cac_gpu_bench
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16gb --time=01:00:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/gpu_bench_%j.log
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
cd $R/test/gpu_bench
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 2; done ) > smi.log 2>&1 &
SMI=$!
for prec in fp64 df64 fp32; do
  echo "=== bench cac/nep/kk one $prec"
  mpirun -np 1 $LMP -in in.bench_cacnep -var ps "cac/nep/kk one $prec" -log log_$prec.lammps > out_$prec.txt 2>&1
  head -12 out_$prec.txt | grep -iE "kokkos|cuda|device|thread|LAMMPS"
  grep -E "Loop time|^ +[0-9]+ " out_$prec.txt | tail -4
done
kill $SMI
echo "--- GPU util samples"; sort smi.log | uniq -c | sort -rn | head -5
echo "=== CPU reference cac/nep (1 rank)"
mpirun -np 1 $LMP -in in.bench_cacnep -var ps "cac/nep one" -log log_cpu.lammps > out_cpu.txt 2>&1
grep -E "Loop time" out_cpu.txt
