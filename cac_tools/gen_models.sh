#!/bin/bash
# generate the jittered validation models (cac data + equivalent full-atom data) used by tools/verify_all.sh
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
cd $R/runs/valclassic
T=$R/tools/make_cac_model.py
gen() { n=$1; shift; [ -f ${n}_full.data ] && return; $PY $T "$@" --jitter 0.04 -o ${n}.data > /dev/null; $PY $R/tools/expand_cac.py ${n}.data ${n}_full.data > /dev/null; }
SIC="--lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4"
CUC="--lattice fcc --a 3.615 --masses 63.546 --blocks 3 3 3 --block 4"
ALN="--lattice wurtzite --a 3.112 --c 4.982 --masses 26.9815 14.007 --blocks 3 2 2 --block 4"
gen si_a $SIC --core 1.0; gen si_e $SIC --core 0.0; gen si_m $SIC --core 0.25
gen cu_a $CUC --core 1.0; gen cu_e $CUC --core 0.0; gen cu_m $CUC --core 0.25
gen si_s3 --lattice diamond --a 5.431 --masses 28.0855 --blocks 4 4 4 --block 3 --core 0.2
gen cu_s3 --lattice fcc --a 3.615 --masses 63.546 --blocks 4 4 4 --block 3 --core 0.2
gen aln_a $ALN --core 1.0; gen aln_e $ALN --core 0.0; gen aln_m $ALN --core 0.25
gen aln_s3 --lattice wurtzite --a 3.112 --c 4.982 --masses 26.9815 14.007 --blocks 4 3 3 --block 3 --core 0.2
gen si_f --lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4 --block2 2 --split -0.1 --core 0.0
gen cu_f --lattice fcc --a 3.615 --masses 63.546 --blocks 3 3 3 --block 4 --block2 2 --split -0.1 --core 0.0
gen si_g --lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4 --block2 2 --split -0.1 --core 0.25
gen cu_g --lattice fcc --a 3.615 --masses 63.546 --blocks 3 3 3 --block 4 --block2 2 --split -0.1 --core 0.25
gen aln_f $ALN --block2 2 --split -0.1 --core 0.0
gen aln_g $ALN --block2 2 --split -0.1 --core 0.25
cp -f $R/test/nep.txt . 
