#!/bin/bash
# final small-model benchmarks (CPU 1 thread, CPU 16 threads, GPU), on the idle allocation
J=${1:-47841126}
cd /blue/ypchen/ntaormina/nep/cac_kokkos
export NS_STATIC=20 NS_REBUILD=5
srun --jobid=$J --overlap env OMP_NUM_THREADS=1 TAG=fin_cpu1 STYLES="eam eamal eamfs lj sw tersoff" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=fin_cpu16 STYLES="eam eamal eamfs lj sw tersoff" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=fin_gpu STYLES="eamkk eamalkk eamfskk ljkk swkk terskk" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=$J --overlap env OMP_NUM_THREADS=16 TAG=fin_nep STYLES="nep nepkk nepkk32" NS_STATIC=10 NS_REBUILD=3 bash tools/bench_classic.sh > /dev/null 2>&1
echo FINAL_BENCH_DONE
