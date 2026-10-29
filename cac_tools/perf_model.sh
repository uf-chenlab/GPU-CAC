#!/bin/bash
# usage: srun --jobid=<id> --overlap bash tools/perf_model.sh <model> <nsteps> <rebuild 0/1> [ps]
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
M=$1; NS=$2; RB=$3; PS=${4:-"cac/nep/kk one fp64"}
D=$R/runs/perf; mkdir -p $D; cd $D; cp $R/test/nep.txt $R/test/in.bench_models $R/models/m_$M.data .
perf record -F 400 -g -o perf_${M}_$RB.data mpirun --bind-to none -np 1 $LMP -in in.bench_models -var data m_$M.data -var ps "$PS" -var nsteps $NS -var rebuild $RB -log none > perf_${M}_$RB.out 2>&1
grep -E "Total wall|Loop time|^Pair|^Neigh" perf_${M}_$RB.out
perf report -i perf_${M}_$RB.data --no-children --sort symbol -g none --stdio 2>/dev/null | grep -v "^#" | grep -v "^$" | head -${TOPN:-18} | cut -c1-130
