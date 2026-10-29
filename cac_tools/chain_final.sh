#!/bin/bash
cd /blue/ypchen/ntaormina/nep/cac_kokkos
# alloc 1: rebuild the canonical build with the final sources, then quick verification of every style
( srun --jobid=47841126 --overlap bash tools/rebuild.sh > logs/rebuild_final.log 2>&1
  for t in sw eam lj ters nep eamal eamfs swkk eamkk ljkk terskk eamalkk eamfskk nepkk; do
    srun --jobid=47841126 --overlap bash tools/verify_all.sh quick $t > logs/vf_$t.log 2>&1
    echo "$t: $(grep -E "^SUMMARY" logs/vf_$t.log) fails=$(grep -c "^FAIL" logs/vf_$t.log) nodal-identical=$(grep -c "nodal forces.*byte-identical" logs/vf_$t.log)"
  done > logs/final_quick.log 2>&1
  echo FINAL_QUICK_DONE >> logs/final_quick.log ) &
# alloc 2: full matrix of the EAM family on the final binary (separate work directory)
( for t in eam eamal eamfs; do
    echo "######## $t"
    srun --jobid=47873940 --overlap env LMP=/blue/ypchen/ntaormina/nep/cac_kokkos/build_cuda2/lmp VDIR=valclassic_c bash tools/verify_all.sh full $t
  done > logs/final_full_eam.log 2>&1
  echo FINAL_EAM_DONE >> logs/final_full_eam.log ) &
# alloc 3: final timings with the final binary, all on the same node
( export LMP=/blue/ypchen/ntaormina/nep/cac_kokkos/build_cuda2/lmp NS_STATIC=20 NS_REBUILD=5
  S6="eam eamal eamfs lj sw tersoff"; K6="eamkk eamalkk eamfskk ljkk swkk terskk"; J=47874600
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=1  TAG=H_cpu1  STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=H_cpu16 STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=H_gpu   STYLES="$K6" bash tools/bench_classic.sh > /dev/null 2>&1
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=H_nep   STYLES="nep nepkk nepkk32" NS_STATIC=10 NS_REBUILD=3 bash tools/bench_classic.sh > /dev/null 2>&1
  export NS_STATIC=10 NS_REBUILD=3
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=H_big_cpu16 MODELS="big_e big_m" STYLES="$S6" bash tools/bench_classic.sh > /dev/null 2>&1
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=H_big_gpu   MODELS="big_e big_m" STYLES="$K6" bash tools/bench_classic.sh > /dev/null 2>&1
  srun --jobid=$J --overlap env LMP=$LMP OMP_NUM_THREADS=16 TAG=H_big_nep   MODELS="big_e big_m" STYLES="nepkk nepkk32" bash tools/bench_classic.sh > /dev/null 2>&1
  echo FINAL_BENCH_DONE > logs/final_bench_h.log ) &
wait
