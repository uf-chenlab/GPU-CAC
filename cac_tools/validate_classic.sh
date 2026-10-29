#!/bin/bash
# usage: srun ... bash tools/validate_classic.sh   -- compares legacy CAC, new CAC and full-atom LAMMPS (energy, 1-step temperature/pressure)
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
D=$R/runs/valclassic; mkdir -p $D; cd $D; cp $R/test_classic/in.classic $R/models_classic/Si.sw $R/models_classic/Si.tersoff $R/models_classic/Cu_u3.eam $R/models_classic/Cu_mishin1.eam.alloy $R/models_classic/CuZr_mm.eam.fs .
T=$R/tools/make_cac_model.py
SIC="--lattice diamond --a 5.431 --masses 28.0855 --blocks 3 3 3 --block 4"
CUC="--lattice fcc --a 3.615 --masses 63.546 --blocks 3 3 3 --block 4"
for m in "si_a:$SIC --core 1.0" "si_e:$SIC --core 0.0" "si_m:$SIC --core 0.25" "cu_a:$CUC --core 1.0" "cu_e:$CUC --core 0.0" "cu_m:$CUC --core 0.25"; do
  n=${m%%:*}; a=${m#*:}; $PY $T $a --jitter 0.04 -o ${n}.data > /dev/null; $PY $R/tools/expand_cac.py ${n}.data ${n}_full.data > /dev/null
done
cat > in.full <<"EOT"
units metal
boundary s s s
atom_style atomic
read_data ${data}
pair_style ${pstyle}
pair_coeff ${pcoeff}
thermo_style custom step pe
thermo_modify format float %24.16e
run 0
EOT
fl() { mpirun --bind-to none -np 1 $LMP "$@" -log none 2>&1; }
printf "%-8s %-6s %-9s %22s %12s %12s\n" model style path PE temp1 press1
run_one() { # model style fullps fullpc cacps cacpc avec
  local n=$1 st=$2
  pe=$(fl -in in.full -var data ${n}_full.data -var pstyle "$3" -var pcoeff "$4" | grep -A1 "^ *Step" | tail -1 | awk '{print $2}')
  printf "%-8s %-6s %-9s %22s\n" $n $st fullatom "$pe"
  for mode in legacy new; do
    [ $mode = legacy ] && export CAC_LEGACY_LISTS=1 || unset CAC_LEGACY_LISTS
    out=$(fl -in in.classic -var data ${n}.data -var avec "$7" -var pstyle "$5" -var pcoeff "$6" -var nsteps 1 -var rebuild 0 | grep -A2 "^ *Step" | tail -2)
    pe0=$(echo "$out" | sed -n 1p | awk '{print $3}'); t1=$(echo "$out" | sed -n 2p | awk '{print $2}'); p1=$(echo "$out" | sed -n 2p | awk '{print $4}')
    printf "%-8s %-6s %-9s %22s %12.5e %12.5e\n" $n $st $mode "$pe0" $t1 $p1
  done
}
for n in si_a si_e si_m; do
  run_one $n sw "sw" "* * Si.sw Si" "cac/sw one" "* * Si.sw Si" "cac 8 8"
  run_one $n ters "tersoff" "* * Si.tersoff Si" "cac/tersoff one" "* * Si.tersoff Si" "cac 8 8"
done
for n in cu_a cu_e cu_m; do
  run_one $n eam "eam" "* * Cu_u3.eam" "cac/eam one" "* * Cu_u3.eam" "cac 8 4"
  run_one $n eamal "eam/alloy" "* * Cu_mishin1.eam.alloy Cu" "cac/eam/alloy one" "* * Cu_mishin1.eam.alloy Cu" "cac 8 4"
  run_one $n lj "lj/cut 5.0" "1 1 0.4 2.3" "cac/lj 5.0 one" "1 1 0.4 2.3" "cac 8 4"
done
