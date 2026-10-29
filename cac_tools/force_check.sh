#!/bin/bash
# usage: bash tools/force_check.sh   (inside the allocation) -- quadrature-point forces of the new site kernels vs full-atom LAMMPS
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd $R/runs/valclassic
cat > in.fdump <<"EOT"
units metal
boundary s s s
atom_style atomic
read_data ${data}
pair_style ${pstyle}
pair_coeff ${pcoeff}
compute pe all pe/atom
dump d all custom 1 ${out} id type x y z fx fy fz c_pe
dump_modify d format float %.17g sort id
thermo_style custom step pe
run 0
EOT
fl() { mpirun --bind-to none -np 1 $LMP "$@" -log none > /dev/null 2>&1; }
chk() { # model tag fullstyle fullcoeff cacstyle caccoeff avec
  unset CAC_LEGACY_LISTS
  fl -in in.fdump -var data $1_full.data -var pstyle "$3" -var pcoeff "$4" -var out full_$2.dump
  CAC_DUMP_QUAD=q_$2 fl -in in.classic -var data $1.data -var avec "$7" -var pstyle "$5" -var pcoeff "$6" -var nsteps 0 -var rebuild 0
  echo "== $1 $2 (new site kernel vs full-atom LAMMPS)"; $PY $R/tools/compare_quad.py q_$2 full_$2.dump 1e-9 2>&1 | grep -E "quad points|force|site energy|ignoring"
}
for n in si_a si_e si_m; do
  chk $n sw   "sw" "* * Si.sw Si" "cac/sw one" "* * Si.sw Si" "cac 8 8"
  chk $n ters "tersoff" "* * Si.tersoff Si" "cac/tersoff one" "* * Si.tersoff Si" "cac 8 8"
done
for n in cu_a cu_e cu_m; do
  chk $n eam   "eam" "* * Cu_u3.eam" "cac/eam one" "* * Cu_u3.eam" "cac 8 4"
  chk $n eamfs "eam/fs" "* * CuZr_mm.eam.fs Cu" "cac/eam/fs one" "* * CuZr_mm.eam.fs Cu" "cac 8 4"
  chk $n lj    "lj/cut 5.0" "1 1 0.4 2.3" "cac/lj 5.0 one" "1 1 0.4 2.3" "cac 8 4"
done
# odd element scale (3x3x3 cells) in a mixed model
T=$R/tools/make_cac_model.py
$PY $T --lattice diamond --a 5.431 --masses 28.0855 --blocks 4 4 4 --block 3 --core 0.2 --jitter 0.04 -o si_s3.data > /dev/null; $PY $R/tools/expand_cac.py si_s3.data si_s3_full.data > /dev/null
$PY $T --lattice fcc --a 3.615 --masses 63.546 --blocks 4 4 4 --block 3 --core 0.2 --jitter 0.04 -o cu_s3.data > /dev/null; $PY $R/tools/expand_cac.py cu_s3.data cu_s3_full.data > /dev/null
chk si_s3 sw   "sw" "* * Si.sw Si" "cac/sw one" "* * Si.sw Si" "cac 8 8"
chk si_s3 ters "tersoff" "* * Si.tersoff Si" "cac/tersoff one" "* * Si.tersoff Si" "cac 8 8"
chk cu_s3 eam  "eam" "* * Cu_u3.eam" "cac/eam one" "* * Cu_u3.eam" "cac 8 4"
chk cu_s3 lj   "lj/cut 5.0" "1 1 0.4 2.3" "cac/lj 5.0 one" "1 1 0.4 2.3" "cac 8 4"
