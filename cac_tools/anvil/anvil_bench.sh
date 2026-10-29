#!/bin/bash
# Portable CAC benchmark: old code path vs new code path, one binary.
#   CODE=new  -> default (new site-only lists + threaded kernels)
#   CODE=old  -> CAC_LEGACY_LISTS=1 (original per-quadrature-point neighbour scan and kernels)
# usage (inside a compute allocation, from the directory that holds the kit):
#   LMP=/path/to/lmp CODE=new NP=1 THREADS=16 STYLES="eam lj" MODELS="cu_big_e cu_big_m" bash anvil_bench.sh
# Styles: eam eamal eamfs lj (Cu models cu_*)   sw tersoff (Si models si_*)
# Output: one line per (style, model, run) in results.tsv, raw logs in out/
set -u
LMP=${LMP:?set LMP}; CODE=${CODE:-new}; NP=${NP:-1}; THREADS=${THREADS:-1}
STYLES=${STYLES:-"eam eamal eamfs lj sw tersoff"}
MODELS=${MODELS:-"cu_big_e cu_big_m si_big_e si_big_m"}
NS_STATIC=${NS_STATIC:-20}; NS_REBUILD=${NS_REBUILD:-3}; TMO=${TMO:-1500}
TAG=${TAG:-${CODE}_np${NP}_t${THREADS}}
mkdir -p out; : > /dev/null
export OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=true OMP_PLACES=cores
[ "$CODE" = old ] && export CAC_LEGACY_LISTS=1 || unset CAC_LEGACY_LISTS
for st in $STYLES; do
  case $st in
    eam)     fam=cu; avec="cac 8 4"; ps="cac/eam one";       pc="* * Cu_u3.eam" ;;
    eamal)   fam=cu; avec="cac 8 4"; ps="cac/eam/alloy one"; pc="* * Cu_mishin1.eam.alloy Cu" ;;
    eamfs)   fam=cu; avec="cac 8 4"; ps="cac/eam/fs one";    pc="* * CuZr_mm.eam.fs Cu" ;;
    lj)      fam=cu; avec="cac 8 4"; ps="cac/lj 5.0 one";    pc="1 1 0.4 2.3" ;;
    sw)      fam=si; avec="cac 8 8"; ps="cac/sw one";        pc="* * Si.sw Si" ;;
    tersoff) fam=si; avec="cac 8 8"; ps="cac/tersoff one";   pc="* * Si.tersoff Si" ;;
  esac
  for m in $MODELS; do
    case $m in ${fam}_*) ;; *) continue ;; esac
    for run in static rebuild; do
      if [ $run = static ]; then ns=$NS_STATIC; rb=0; else ns=$NS_REBUILD; rb=1; fi
      f=out/${TAG}_${st}_${m}_$run.out
      # memory sampler: every 0.5 s sum and max of the resident set of all running lmp ranks (kB)
      ( peak_sum=0; peak_max=0
        while sleep 0.5; do
          r=$(ps -C $(basename $LMP) -o rss= 2>/dev/null | awk '{s+=$1; if($1>m)m=$1} END{print s+0, m+0}')
          set -- $r; [ "$1" -gt "$peak_sum" ] && peak_sum=$1; [ "$2" -gt "$peak_max" ] && peak_max=$2
          echo "$peak_sum $peak_max" > $f.mem
        done ) &
      samp=$!
      timeout $TMO mpirun --bind-to none -np $NP $LMP -in in.classic -var data $m.data -var avec "$avec" \
        -var pstyle "$ps" -var pcoeff "$pc" -var nsteps $ns -var rebuild $rb -log none > $f 2>&1
      kill $samp 2>/dev/null; wait $samp 2>/dev/null
      mem=$(cat $f.mem 2>/dev/null || echo "0 0"); memsum=$(( ${mem% *} / 1024 )); memmax=$(( ${mem#*} / 1024 ))
      lmpmem=$(awk '/Per MPI rank memory allocation/{getline; getline; print $0}' $f | head -1)
      loop=$(awk '/Loop time/{print $4}' $f); wall=$(awk '/Total wall/{print $4}' $f)
      pe0=$(awk '/^ +0 /{print $3}' $f | head -1); pe1=$(awk -v n=$ns '$1==n{print $3}' $f | tail -1)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $TAG $st $m $run $ns "${loop:-FAIL}" "${wall:-}" "${pe0:-}" "${pe1:-}" "$NP x $THREADS" "$memsum" "$memmax" >> results.tsv
      echo "$TAG $st $m $run loop=${loop:-FAIL} wall=${wall:-}"
    done
  done
done
# results.tsv columns: tag style model run nsteps looptime_s walltime_s pe_step0 pe_last ranksxthreads peak_total_RSS_MB peak_max_rank_RSS_MB
# per-step time (ms)  = 1000*loop/nsteps of the 'static' run
# rebuild cost (ms)   = 1000*loop/nsteps of the 'rebuild' run (rebuild every step) minus the per-step time
# setup time (s)      = wall - loop (includes reading the data file and the first neighbour build)
