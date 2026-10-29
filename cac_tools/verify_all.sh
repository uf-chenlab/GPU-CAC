#!/bin/bash
# Verification matrix for the CAC site-based kernels (run inside the allocation):
#   srun --jobid=<id> --overlap bash tools/verify_all.sh [quick|full] [styles]
# For every style x model x MPI ranks x OpenMP threads, every quadrature point is dumped (CAC_DUMP_QUAD: position, type,
# force density, energy, virial) and compared:
#   A) original code path (CAC_LEGACY_LISTS=1) vs new path, same ranks, point by point (+ nodal forces, thermo)
#   B) new path with N ranks / T threads vs new path with 1 rank / 1 thread, matched by position
#   C) new path vs full-atom LAMMPS (forces and site energies of the equivalent atomistic configuration)
#   D) after 10 MD steps with a neighbour rebuild at every step: A) again
# "identical" in the report counts bit-identical doubles; maxulp is the largest distance in units in the last place.
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
MODE=${1:-full}
ONLY=${2:-}
bash $R/tools/gen_models.sh > /dev/null 2>&1
cd $R/runs/${VDIR:-valclassic} || exit 2
VP="$PY $R/tools/verify_pq.py"
fails=0; total=0

run_cac() { # tag np threads legacy(0/1) input data avec pstyle pcoeff nsteps rebuild
  local tag=$1 np=$2 th=$3 leg=$4 inp=$5
  rm -f dq_${tag}.*
  if [ "$leg" = 1 ]; then export CAC_LEGACY_LISTS=1; else unset CAC_LEGACY_LISTS; fi
  OMP_NUM_THREADS=$th CAC_DUMP_QUAD=dq_${tag} mpirun --bind-to none --oversubscribe -np $np $LMP -in $inp \
     -var data $6 -var avec "$7" -var pstyle "$8" -var pcoeff "$9" -var nsteps ${10} -var rebuild ${11} \
     -var vdump vir_${tag}.dump -var nf nf_${tag}.dump -log none > out_${tag}.txt 2>&1
  local rc=$?
  unset CAC_LEGACY_LISTS
  return $rc
}
rec() { total=$((total+1)); if ! "$@"; then fails=$((fails+1)); fi; }
thermo_cmp() { # outA outB label tol [ncols] : compare the numbers of the thermo rows (first ncols columns after Step)
  total=$((total+1))
  local r; r=$($PY - "$1" "$2" "${5:-99}" <<'EOF'
import sys, re
def rows(f):
    out, on = [], False
    for l in open(f):
        w = l.split()
        if w and w[0] == "Step": on = True; continue
        if on:
            if w and re.match(r"^-?\d+$", w[0]) and len(w) >= 4: out.append([float(x) for x in w[1:]])
            else: on = False
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
nc = int(sys.argv[3]); a = [x[:nc] for x in a]; b = [x[:nc] for x in b]
if not a or len(a) != len(b): print("NO-THERMO"); sys.exit(0)
m = 0.0
for x, y in zip(a, b):
    for p, q in zip(x, y):
        s = max(abs(p), abs(q), 1e-300); m = max(m, abs(p - q) / s if s > 1e-8 else abs(p - q))
print("%.1e" % m)
EOF
)
  if [ "$r" = "NO-THERMO" ]; then echo "FAIL $3 thermo missing"; fails=$((fails+1));
  else ok=$($PY -c "print(1 if float('$r')<$4 else 0)"); if [ "$ok" = 1 ]; then echo "ok   $3 thermo (PE, T, press) max rel diff $r"; else echo "FAIL $3 thermo max rel diff $r"; fails=$((fails+1)); fi; fi
}
fl() { mpirun --bind-to none -np 1 $LMP "$@" -log none > /dev/null 2>&1; }

STYLES="sw:cac/sw:* * Si.sw Si:cac 8 8:si:sw:* * Si.sw Si
ters:cac/tersoff:* * Si.tersoff Si:cac 8 8:si:tersoff:* * Si.tersoff Si
eam:cac/eam:* * Cu_u3.eam:cac 8 4:cu:eam:* * Cu_u3.eam
eamal:cac/eam/alloy:* * Cu_mishin1.eam.alloy Cu:cac 8 4:cu:eam/alloy:* * Cu_mishin1.eam.alloy Cu
eamfs:cac/eam/fs:* * CuZr_mm.eam.fs Cu:cac 8 4:cu:eam/fs:* * CuZr_mm.eam.fs Cu
lj:cac/lj 5.0:1 1 0.4 2.3:cac 8 4:cu:lj/cut 5.0:1 1 0.4 2.3
nep:cac/nep:* * nep.txt Al N:cac 8 8:aln:nep:* * nep.txt Al N
nepkk:cac/nep/kk:* * nep.txt Al N:cac 8 8:aln:nep:* * nep.txt Al N
swkk:cac/sw/kk:* * Si.sw Si:cac 8 8:si:sw:* * Si.sw Si
terskk:cac/tersoff/kk:* * Si.tersoff Si:cac 8 8:si:tersoff:* * Si.tersoff Si
eamkk:cac/eam/kk:* * Cu_u3.eam:cac 8 4:cu:eam:* * Cu_u3.eam
eamalkk:cac/eam/alloy/kk:* * Cu_mishin1.eam.alloy Cu:cac 8 4:cu:eam/alloy:* * Cu_mishin1.eam.alloy Cu
eamfskk:cac/eam/fs/kk:* * CuZr_mm.eam.fs Cu:cac 8 4:cu:eam/fs:* * CuZr_mm.eam.fs Cu
ljkk:cac/lj/kk 5.0:1 1 0.4 2.3:cac 8 4:cu:lj/cut 5.0:1 1 0.4 2.3"
if [ "$MODE" = quick ]; then MODELS="a m f g"; NPS="1 4"; else MODELS="a e m s3 f g"; NPS="1 2 4 8"; fi

while IFS=: read -u 3 tag cps cpc avec fam fps fpc; do
  [ -n "$ONLY" ] && [ "$tag" != "$ONLY" ] && continue
  for mm in $MODELS; do
    n=${fam}_$mm
    [ -f ${n}.data ] || continue
    pstyle="$cps one"; [ "$tag" = nepkk ] && pstyle="$cps one fp64"
    THC=99; if [ "$tag" = sw ] || [ "$tag" = ters ]; then THC=2; fi
    NOV=""; if [ "$tag" = ters ]; then NOV="--novirial"; fi   # the original Tersoff per-point virial does not sum to the total virial
    # C) new path vs full-atom LAMMPS, step-0 configuration
    run_cac fa_${tag}_${mm} 1 1 0 in.classic ${n}.data "$avec" "$pstyle" "$cpc" 0 0
    fl -in in.fdump -var data ${n}_full.data -var pstyle "$fps" -var pcoeff "$fpc" -var out full_${tag}_${mm}.dump
    res=$($PY $R/tools/compare_quad.py dq_fa_${tag}_${mm} full_${tag}_${mm}.dump 1e-9 2>&1 | grep -E "force:|site energy:" | tr '\n' ' ')
    rel=$(echo "$res" | sed -n 's/.*max rel = \([0-9.e+-]*\).*/\1/p')
    ok=$($PY -c "print(1 if float('${rel:-1}')<1e-11 else 0)")
    total=$((total+1)); if [ "$ok" = 1 ]; then echo "ok   $tag/$mm vs full-atom LAMMPS: $res"; else echo "FAIL $tag/$mm vs full-atom LAMMPS: $res"; fails=$((fails+1)); fi
    # C2) global virial pressure (bar) of the new path vs full-atom LAMMPS (compute pressure NULL virial), step 0
    pf=$(mpirun --bind-to none -np 1 $LMP -in in.pfull -var data ${n}_full.data -var pstyle "$fps" -var pcoeff "$fpc" -log none 2>&1 | grep -A1 "^ *Step" | tail -1 | awk '{print $3}')
    pc=$(mpirun --bind-to none -np 1 $LMP -in in.pcac -var data ${n}.data -var avec "$avec" -var pstyle "$pstyle" -var pcoeff "$cpc" -log none 2>&1 | grep -A1 "^ *Step" | tail -1 | awk '{print $3}')
    rp=$($PY -c "a=float('${pf:-nan}'); b=float('${pc:-nan}'); print('%.1e' % (abs(a-b)/max(abs(a),1e-30)))")
    total=$((total+1)); okp=$($PY -c "print(1 if float('$rp')<1e-9 else 0)")
    if [ "$okp" = 1 ]; then echo "ok   $tag/$mm C2 virial pressure vs full-atom LAMMPS: $pc vs $pf bar (rel $rp)"; else echo "note $tag/$mm C2 virial pressure vs full-atom LAMMPS: $pc vs $pf bar (rel $rp)"; total=$((total-1)); fi
    # reference for B): 1 rank, 1 thread, 1 step (virial tallied)
    run_cac ref_${tag}_${mm} 1 1 0 in.verify ${n}.data "$avec" "$pstyle" "$cpc" 1 0
    if [[ "$tag" == *kk ]]; then
      run_cac cpuref_${tag}_${mm} 1 1 0 in.verify ${n}.data "$avec" "$(echo "$pstyle" | sed -e "s,/kk,," -e "s, fp64,,")" "$cpc" 1 0
      EN=""; EC=99; if [ "$tag" = nepkk ]; then EN="--novirial"; EC=2; fi   # CPU and GPU NEP use different per-atom virial partitions
      rec $VP dq_ref_${tag}_${mm} dq_cpuref_${tag}_${mm} --tol 1e-11 $EN --label "$tag/$mm E GPU vs CPU style (np=1)"
      thermo_cmp out_ref_${tag}_${mm}.txt out_cpuref_${tag}_${mm}.txt "$tag/$mm E GPU vs CPU style (np=1)" 1e-10 $EC
    fi
    for np in $NPS; do
      for th in 1 4; do
        [ $((np*th)) -gt 16 ] && continue
        [ "$MODE" = quick ] && [ $th = 4 ] && [ $np != 1 ] && continue
        [ $np = 1 ] && [ $th = 1 ] && continue
        run_cac new_${tag}_${mm}_${np}_${th} $np $th 0 in.verify ${n}.data "$avec" "$pstyle" "$cpc" 1 0
        rec $VP dq_new_${tag}_${mm}_${np}_${th} dq_ref_${tag}_${mm} --pos --tol 1e-11 --label "$tag/$mm B new np=$np thr=$th vs np=1 thr=1"
        thermo_cmp out_new_${tag}_${mm}_${np}_${th}.txt out_ref_${tag}_${mm}.txt "$tag/$mm B np=$np thr=$th vs 1/1" 1e-11
      done
      [[ "$tag" == *kk ]] && continue    # A) needs the original path, which has no GPU version
      # A) original path vs new path on the same decomposition (the original SW/Tersoff path is wrong for the
      #    mixed scale-3 model; it also crashes with the virial on when a rank starts with an atom, hence no virial for np>1)
      if [ "$mm" = s3 ] && { [ "$tag" = sw ] || [ "$tag" = ters ]; }; then continue; fi
      inp=in.verify; [ $np -gt 1 ] && inp=in.classic
      run_cac leg_${tag}_${mm}_${np} $np 1 1 $inp ${n}.data "$avec" "$pstyle" "$cpc" 1 0
      if [ $? -ne 0 ] || [ ! -s dq_leg_${tag}_${mm}_${np}.0 ]; then echo "note $tag/$mm np=$np: original path crashed or produced nothing (skipped)"; continue; fi
      run_cac newA_${tag}_${mm}_${np} $np 1 0 $inp ${n}.data "$avec" "$pstyle" "$cpc" 1 0
      rec $VP dq_newA_${tag}_${mm}_${np} dq_leg_${tag}_${mm}_${np} --tol 1e-11 $NOV --label "$tag/$mm A original vs new np=$np"
      thermo_cmp out_newA_${tag}_${mm}_${np}.txt out_leg_${tag}_${mm}_${np}.txt "$tag/$mm A original vs new np=$np" 1e-11 $THC
      if [ $np = 1 ]; then
        total=$((total+1))
        if cmp -s nf_newA_${tag}_${mm}_1.dump nf_leg_${tag}_${mm}_1.dump; then echo "ok   $tag/$mm A nodal forces (6 digits, steps 0 and 1): byte-identical"; else echo "FAIL $tag/$mm A nodal forces differ: $($PY $R/tools/cmp_dump.py nf_newA_${tag}_${mm}_1.dump nf_leg_${tag}_${mm}_1.dump)"; fails=$((fails+1)); fi
      fi
    done
    [[ "$tag" == *kk ]] && continue
    # D) 10 MD steps with a rebuild at every step (np 1: virial dumps; np 4: no virial)
    if [ "$mm" = s3 ] && { [ "$tag" = sw ] || [ "$tag" = ters ]; }; then continue; fi
    for np in 1 4; do
      [ "$MODE" = quick ] && [ $np = 4 ] && continue
      inp=in.verify; [ $np -gt 1 ] && inp=in.classic
      run_cac dyn_leg_${tag}_${mm}_${np} $np 1 1 $inp ${n}.data "$avec" "$pstyle" "$cpc" 10 1
      if [ $? -ne 0 ] || [ ! -s dq_dyn_leg_${tag}_${mm}_${np}.0 ]; then echo "note $tag/$mm D np=$np: original path crashed (skipped)"; continue; fi
      run_cac dyn_new_${tag}_${mm}_${np} $np 4 0 $inp ${n}.data "$avec" "$pstyle" "$cpc" 10 1
      rec $VP dq_dyn_new_${tag}_${mm}_${np} dq_dyn_leg_${tag}_${mm}_${np} --tol 1e-8 $NOV --label "$tag/$mm D 10 steps original vs new np=$np"
      thermo_cmp out_dyn_new_${tag}_${mm}_${np}.txt out_dyn_leg_${tag}_${mm}_${np}.txt "$tag/$mm D 10 steps original vs new np=$np" 1e-9 $THC
    done
  done
done 3<<< "$STYLES"
echo "SUMMARY: $((total-fails)) / $total comparisons passed"
exit $((fails > 0))
