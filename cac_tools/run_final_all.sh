#!/bin/bash
J=${1:-47841126}
cd /blue/ypchen/ntaormina/nep/cac_kokkos
export NS_STATIC=20 NS_REBUILD=5
S6="eam eamal eamfs lj sw tersoff"; K6="eamkk eamalkk eamfskk ljkk swkk terskk"
srun --jobid=$J --overlap env OMP_NUM_THREADS=1  TAG=F_cpu1  STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=F_cpu16 STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=F_gpu   STYLES="$K6" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=F_nep   STYLES="nep nepkk nepkk32" NS_STATIC=10 NS_REBUILD=3 bash tools/bench_classic.sh > /dev/null 2>&1
export NS_STATIC=10 NS_REBUILD=3
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=F_big_cpu16 MODELS="big_e big_m" STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=F_big_gpu   MODELS="big_e big_m" STYLES="$K6" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=F_big_nep   MODELS="big_e big_m" STYLES="nepkk nepkk32" bash tools/bench_classic.sh > /dev/null 2>&1
echo FINAL_ALL_DONE
