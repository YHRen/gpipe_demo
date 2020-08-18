#!/bin/bash 

bsz=(256 512 1024)
csz=(1 2 4)
dsz=$( echo "2^11" | bc )
fcw=$( echo "2^12" | bc )
fcl=30
epc=4

echo "fcw fcl dsz" $fcw $fcl $dsz

n="${#bsz}"
for (( i=0; i<n; i++ )); do
    b="${bsz[i]}"
    c="${csz[i]}"
    perf_fp="report_bsz_${b}_csz_${c}_dsz_${dsz}.qdrep"
    echo "perf filename " "$perf_fp"
    #nsys profile -t nvtx,cuda -s cpu -o $perf_fp \
    python main.py -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc
done
