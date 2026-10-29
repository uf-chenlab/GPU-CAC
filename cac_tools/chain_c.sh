#!/bin/bash
# allocation 3: timing of the sorted-list build, then quick verification of it
cd /blue/ypchen/ntaormina/nep/cac_kokkos
export LMP=/blue/ypchen/ntaormina/nep/cac_kokkos/build_cuda2/lmp NS_STATIC=20 NS_REBUILD=5
S6="eam eamal eamfs lj sw tersoff"
srun --jobid=47874600 --overlap env LMP=$LMP OMP_NUM_THREADS=1  TAG=G_cpu1  STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
srun --jobid=47874600 --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=G_cpu16 STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
export NS_STATIC=10 NS_REBUILD=3
srun --jobid=47874600 --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=G_big_cpu16 MODELS="big_e big_m" STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
echo TIMING_C_DONE
for t in sw eam lj ters nep eamal eamfs swkk eamkk ljkk terskk eamalkk eamfskk nepkk; do
  srun --jobid=47874600 --overlap env LMP=$LMP VDIR=valclassic_c bash tools/verify_all.sh quick $t > logs/vc_$t.log 2>&1
  echo "$t: $(grep -E "^SUMMARY" logs/vc_$t.log) fails=$(grep -c "^FAIL" logs/vc_$t.log) nodal-identical=$(grep -c "nodal forces.*byte-identical" logs/vc_$t.log)"
done
echo CHAIN_C_DONE
