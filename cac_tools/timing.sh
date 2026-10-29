#!/bin/bash
# CAC_TIMING phase breakdown. usage: srun --jobid=<id> --overlap bash tools/timing.sh "<models>" "<ps>"
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
MODELS=${1:-"atoms elem4 mix4"}; PS=${2:-"cac/nep/kk one fp64"}
D=$R/runs/timing; mkdir -p $D; cd $D; cp $R/test/nep.txt $R/test/in.bench_models .
export CAC_TIMING=1
for m in $MODELS; do cp $R/models/m_$m.data .
  for run in "static 20 0" "rebuild 5 1"; do set -- $run
    mpirun --bind-to none -np 1 $LMP -in in.bench_models -var data m_$m.data -var ps "$PS" -var nsteps $2 -var rebuild $3 -log none > t_${m}_$1.out 2>&1
    echo "$m $1 ($2 steps): $(grep "Loop time" t_${m}_$1.out | awk "{print \$4}") s"; grep CAC_TIMING t_${m}_$1.out
  done
done
