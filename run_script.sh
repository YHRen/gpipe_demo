#!/bin/bash 

mod="${1:-fc}" # nn arch
csz=(1 2 16) # chunk sizes
dsz=$( echo "2^11" | bc ) # data size
epc=10 # epoches

echo $mod

if [[ "$mod" == "fc" ]]; then
    bsz=(512) # batch sizes
    fcw=$( echo "2^13" | bc ) # fc model width
    fcl=20 # fc model length per gpu
    echo "dsz fcw fcl dsz" $dsz $fcw $fcl $dsz
    for b in "${bsz[@]}" ; do
        for c in "${csz[@]}" ; do
            perf_fp="report_mod_${mod}_bsz_${b}_csz_${c}_dsz_${dsz}"
            if [[ $perf == "on" ]]; then
                nsys profile -t nvtx,cuda -s cpu -o $perf_fp \
                python main.py -m $mod -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc
            else
                python main.py -m $mod -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc
            fi
        done
    done
elif [[ "$mod" == "cnn" ]]; then
    bsz=(512) # batch sizes
    fcw=128 # cnn channel size
    fcl=20  # num of cnn layers per gpu
    echo "dsz fcw fcl dsz" $dsz $fcw $fcl $dsz
    for b in "${bsz[@]}"; do
        for c in "${csz[@]}"; do
            perf_fp="report_mod_${mod}_bsz_${b}_csz_${c}_dsz_${dsz}"
            echo "perf filename " "$perf_fp"
            if [[ $perf == "on" ]]; then
                nsys profile -t nvtx,cuda -s cpu -o $perf_fp \
                python main.py -m $mod -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc
            else
                python main.py -m $mod -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc
            fi
        done
    done
fi

