#!/bin/bash
# allocation 2: the CPU-style verification matrix
cd /blue/ypchen/ntaormina/nep/cac_kokkos
for t in sw eam lj nep ters eamal eamfs; do
  echo "######## $t"
  srun --jobid=47873940 --overlap bash tools/verify_all.sh full $t
done > logs/verify_final_cpu.log 2>&1
echo CHAIN_B_DONE >> logs/verify_final_cpu.log
