#!/bin/bash 

bsz=(512)
csz=(1 2 16)
dsz=$( echo "2^11" | bc )
fcw=$( echo "2^13" | bc )
fcl=20
epc=10

echo "fcw fcl dsz" $fcw $fcl $dsz

for b in ${bsz[@]}; do
  for c in ${csz[@]}; do
    perf_fp="report_bsz_${b}_csz_${c}_dsz_${dsz}.qdrep"
    echo "perf filename " "$perf_fp"
    #nsys profile -t nvtx,cuda -s cpu -o $perf_fp \
    python main.py -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc
  done
done
