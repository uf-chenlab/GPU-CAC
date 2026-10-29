#!/bin/bash
# usage: srun --jobid=<id> --overlap env TAG=x STYLES="eam lj" MODELS="atoms elem4 elem8 mix4" NP=1 bash tools/bench_classic.sh
source /blue/ypchen/ntaormina/nep/cac_kokkos/tools/dev_env.sh
TAG=${TAG:-classic}; NP=${NP:-1}; STYLES=${STYLES:-"eam eamal eamfs lj sw tersoff"}; SIZE=${SIZE:-}; MODELS=${MODELS:-"atoms elem4 elem8 mix4"}
NS_STATIC=${NS_STATIC:-5}; NS_REBUILD=${NS_REBUILD:-3}
D=$R/runs/$TAG; mkdir -p $D; cd $D; cp $R/test_classic/in.classic $R/models_classic/* .; cp $R/test/nep.txt .; cp $R/models/m_*.data .; for m in atoms elem4 elem8 mix4; do cp m_$m.data aln_$m.data; done
printf "%-8s %-6s %-8s %9s %9s %9s %8s\n" style model run loop pair neigh wall
for st in $STYLES; do
  case $st in
    eam)     fam=cu; avec="cac 8 4"; ps="cac/eam one";           pc="* * Cu_u3.eam" ;;
    eamal)   fam=cu; avec="cac 8 4"; ps="cac/eam/alloy one";     pc="* * Cu_mishin1.eam.alloy Cu" ;;
    eamfs)   fam=cu; avec="cac 8 4"; ps="cac/eam/fs one";        pc="* * CuZr_mm.eam.fs Cu" ;;
    lj)      fam=cu; avec="cac 8 4"; ps="cac/lj 5.0 one";        pc="1 1 0.4 2.3" ;;
    sw)      fam=si; avec="cac 8 8"; ps="cac/sw one";            pc="* * Si.sw Si" ;;
    tersoff) fam=si; avec="cac 8 8"; ps="cac/tersoff one";       pc="* * Si.tersoff Si" ;;
    eamkk)     fam=cu; avec="cac 8 4"; ps="cac/eam/kk one";           pc="* * Cu_u3.eam" ;;
    eamalkk)   fam=cu; avec="cac 8 4"; ps="cac/eam/alloy/kk one";     pc="* * Cu_mishin1.eam.alloy Cu" ;;
    eamfskk)   fam=cu; avec="cac 8 4"; ps="cac/eam/fs/kk one";        pc="* * CuZr_mm.eam.fs Cu" ;;
    ljkk)      fam=cu; avec="cac 8 4"; ps="cac/lj/kk 5.0 one";        pc="1 1 0.4 2.3" ;;
    swkk)      fam=si; avec="cac 8 8"; ps="cac/sw/kk one";            pc="* * Si.sw Si" ;;
    terskk)    fam=si; avec="cac 8 8"; ps="cac/tersoff/kk one";       pc="* * Si.tersoff Si" ;;
    nep)       fam=aln; avec="cac 8 8"; ps="cac/nep one";             pc="* * nep.txt Al N" ;;
    nepkk)     fam=aln; avec="cac 8 8"; ps="cac/nep/kk one fp64";     pc="* * nep.txt Al N" ;;
    nepkk32)   fam=aln; avec="cac 8 8"; ps="cac/nep/kk one fp32";     pc="* * nep.txt Al N" ;;
  esac
  for m in $MODELS; do
    [ $fam = si ] && [ $m = elem8 ] && mm=elem6 || mm=$m
    for run in static rebuild; do
      if [ $run = static ]; then ns=$NS_STATIC; rb=0; else ns=$NS_REBUILD; rb=1; fi
      f=o_${st}_${m}_$run.out
      timeout ${TMO:-900} mpirun --bind-to none -np $NP $LMP -in in.classic -var data ${fam}_$mm.data -var avec "$avec" -var pstyle "$ps" -var pcoeff "$pc" -var nsteps $ns -var rebuild $rb -log none > $f 2>&1
      loop=$(awk '/Loop time/{print $4}' $f); pair=$(awk -F'|' '/^Pair/{gsub(/ /,"",$3);print $3}' $f)
      neigh=$(awk -F'|' '/^Neigh/{gsub(/ /,"",$3);print $3}' $f); wall=$(awk '/Total wall/{print $4}' $f)
      printf "%-8s %-6s %-8s %9s %9s %9s %8s\n" $st $m $run "${loop:-FAIL}" "$pair" "$neigh" "$wall"
    done
  done
done
