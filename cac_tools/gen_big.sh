#!/bin/bash
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
cd $R/models_classic
T=$R/tools/make_cac_model.py
CU="--lattice fcc --a 3.615 --masses 63.546"
SI="--lattice diamond --a 5.431 --masses 28.0855"
AL="--lattice wurtzite --a 3.112 --c 4.982 --masses 26.9815 14.007"
$PY $T $CU --blocks 8 8 8 --block 8 --core 0.0  -o cu_big_e.data | tail -1
$PY $T $CU --blocks 8 8 8 --block 8 --core 0.25 -o cu_big_m.data | tail -1
$PY $T $SI --blocks 6 6 6 --block 8 --core 0.0  -o si_big_e.data | tail -1
$PY $T $SI --blocks 6 6 6 --block 8 --core 0.25 -o si_big_m.data | tail -1
$PY $T $AL --blocks 6 4 4 --block 8 --core 0.0  -o aln_big_e.data | tail -1
$PY $T $AL --blocks 6 4 4 --block 8 --core 0.25 -o aln_big_m.data | tail -1
ls -la *big*
