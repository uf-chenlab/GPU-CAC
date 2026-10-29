#!/bin/bash
#SBATCH --job-name=cac_test
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=32gb --time=00:40:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/test_%j.log
# usage: sbatch tools/test_gpu.sbatch [tag] ; runs V1 validation, B1 benchmark (new vs CAC_LEGACY_LISTS)
module purge; module load gcc; module load openmpi/5.0.10; module load cuda/12.8.1; module load conda
source activate neptools_env
export LD_PRELOAD=/blue/ypchen/ntaormina/bin/miniconda38/envs/neptools_env/lib/libstdc++.so.6
export PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_TLS=^ib
R=/blue/ypchen/ntaormina/nep/cac_kokkos; LMP=${LMP:-$R/build_cuda/lmp}; TAG=${1:-run}
D=$R/runs/$TAG; mkdir -p $D; cd $D
cp $R/test/aln_mix.data $R/test/aln_bench.data $R/test/nep.txt $R/test/in.cacnep_v $R/test/in.bench_cacnep .
nvidia-smi -L
for mode in new legacy; do
  [ $mode = legacy ] && export CAC_LEGACY_LISTS=1 || unset CAC_LEGACY_LISTS
  echo "##### V1 validation ($mode)"
  CAC_DUMP_QUAD=q_$mode mpirun -np 1 $LMP -in in.cacnep_v -var ps "cac/nep/kk one fp64" -log none > v_$mode.out 2>&1
  python $R/tools/compare_quad.py q_$mode $R/test/full_forces.dump | head -3
  grep "CAC site table" v_$mode.out
  for prec in fp64 fp32; do
    [ $mode = legacy ] && [ $prec = fp32 ] && continue
    echo "##### B1 bench ($mode, $prec)"
    mpirun -np 1 $LMP -in in.bench_cacnep -var ps "cac/nep/kk one $prec" -log none > b_${mode}_$prec.out 2>&1
    grep -E "CAC site table|Loop time|Total wall|^Pair|^Neigh|^Other|^Modify|Peak|allocation" b_${mode}_$prec.out
  done
done
