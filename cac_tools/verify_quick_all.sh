#!/bin/bash
cd /blue/ypchen/ntaormina/nep/cac_kokkos
for t in sw eam lj ters nep eamal eamfs swkk eamkk ljkk terskk eamalkk eamfskk nepkk; do
  srun --jobid=47841126 --overlap bash tools/verify_all.sh quick $t > logs/vq_$t.log 2>&1
  echo "$t: $(grep -E "^SUMMARY" logs/vq_$t.log) fails=$(grep -c "^FAIL" logs/vq_$t.log) nodal-identical=$(grep -c "nodal forces.*byte-identical" logs/vq_$t.log)"
done
echo VQ_DONE
