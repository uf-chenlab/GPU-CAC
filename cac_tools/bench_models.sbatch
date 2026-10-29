#!/bin/bash
#SBATCH --job-name=cac_models
#SBATCH --account=phillpot
#SBATCH --qos=phillpot
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=48gb --time=01:30:00
#SBATCH --output=/blue/ypchen/ntaormina/nep/cac_kokkos/logs/models_%j.log
# usage: sbatch --export=ALL,TAG=x,MODES="new legacy",MODELS="atoms elem4 elem8 mix4",PS="cac/nep/kk one fp64" tools/bench_models.sbatch
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
export PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/14.2.0/openmpi/5.0.10/lib:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_TLS=^ib
R=/blue/ypchen/ntaormina/nep/cac_kokkos; LMP=${LMP:-$R/build_cuda/lmp}; TAG=${TAG:-run}
MODES=${MODES:-new}; MODELS=${MODELS:-"atoms elem4 elem8 mix4"}; PS=${PS:-"cac/nep/kk one fp64"}
D=$R/runs/$TAG; mkdir -p $D; cd $D; cp $R/test/nep.txt $R/test/in.bench_models .
nvidia-smi -L
printf "%-8s %-7s %-8s %9s %9s %9s %9s %9s\n" model mode run loop pair neigh wall sites
for m in $MODELS; do cp $R/models/m_$m.data .; for mode in $MODES; do
  [ $mode = legacy ] && export CAC_LEGACY_LISTS=1 || unset CAC_LEGACY_LISTS
  for run in static rebuild; do
    if [ $run = static ]; then ns=20; rb=0; else ns=5; rb=1; fi
    mpirun --bind-to none -np 1 $LMP -in in.bench_models -var data m_$m.data -var ps "$PS" -var nsteps $ns -var rebuild $rb -log none > o_${m}_${mode}_$run.out 2>&1
    f=o_${m}_${mode}_$run.out
    loop=$(awk "/Loop time/{print \$4}" $f); pair=$(awk -F"|" "/^Pair/{gsub(/ /,\"\",\$3);print \$3}" $f)
    neigh=$(awk -F"|" "/^Neigh/{gsub(/ /,\"\",\$3);print \$3}" $f); wall=$(awk "/Total wall/{print \$4}" $f)
    sites=$(awk "/unique sites/{print \$7}" $f)
    printf "%-8s %-7s %-8s %9s %9s %9s %9s %9s\n" $m $mode $run "$loop" "$pair" "$neigh" "$wall" "$sites"
  done
done; done
