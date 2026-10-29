#!/bin/bash
# allocation 1: final benchmarks, then the GPU-style verification matrix
cd /blue/ypchen/ntaormina/nep/cac_kokkos
bash tools/run_final_all.sh 47841126 > logs/final_all2.log 2>&1
for t in swkk eamkk ljkk terskk eamalkk eamfskk nepkk; do
  echo "######## $t"
  srun --jobid=47841126 --overlap bash tools/verify_all.sh full $t
done > logs/verify_final_gpu.log 2>&1
echo CHAIN_A_DONE >> logs/verify_final_gpu.log
