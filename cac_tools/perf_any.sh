#!/bin/bash
# usage: perf_any.sh <data> "<avec>" "<pstyle>" "<pcoeff>" <nsteps> <rebuild> [topn]
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
D=$R/runs/perf_any; mkdir -p $D; cd $D; cp $R/test_classic/in.classic $R/models_classic/* . ; cp $R/test/nep.txt .
export CAC_TIMING=${CAC_TIMING:-}
perf record -F 150 -g -o p.data mpirun --bind-to none -np 1 $LMP -in in.classic -var data $1 -var avec "$2" -var pstyle "$3" -var pcoeff "$4" -var nsteps $5 -var rebuild $6 -log none > p.out 2>&1
grep -E "Total wall|Loop time|^Pair|^Neigh|^Modify|^Other|CAC site|ERROR|Memory|allocation" p.out
perf report -i p.data --no-children --sort symbol -g none --stdio 2>/dev/null | grep -v "^#" | grep -v "^$" | head -${7:-25} | cut -c1-150
