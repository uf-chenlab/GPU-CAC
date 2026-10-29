#!/bin/bash
# V1: quadrature forces vs full-atom NEP.  usage: srun --jobid=<id> --overlap bash tools/validate.sh <tag> [ps]
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
TAG=${1:-v}; PS=${2:-"cac/nep/kk one fp64"}
D=$R/runs/$TAG; mkdir -p $D; cd $D; cp $R/test/aln_mix.data $R/test/nep.txt $R/test/in.cacnep_v .
for mode in new legacy; do
  [ $mode = legacy ] && export CAC_LEGACY_LISTS=1 || unset CAC_LEGACY_LISTS
  CAC_DUMP_QUAD=q_$mode mpirun --bind-to none -np 1 $LMP -in in.cacnep_v -var ps "$PS" -log none > v_$mode.out 2>&1
  echo "--- V1 $mode: $PS"; $PY $R/tools/compare_quad.py q_$mode $R/test/full_forces.dump | sed -n 1,3p; grep "CAC site table" v_$mode.out
done
