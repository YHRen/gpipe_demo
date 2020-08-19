#!/bin/bash

nnodes=${1:-1}               # how many nodes in this computation
node_rank=${2:-0}            # current node rank, change to node_rank=1 for the second node.
nproc_per_node=2             # number of models per node
master_addr=${3:-localhost}  # you should use either an ip address,
                             #   or a node name on hpc environment.
port=18888                   # for example, 8888

module load ibm-wml-ce nsight-systems
mod="cnn"
b=32
c=4
dsz=2048
fcw=128
fcl=5
epc=2

python -m torch.distributed.launch \
    --nproc_per_node ${nproc_per_node} \
    --nnodes "$nnodes" \
    --node_rank "$node_rank" \
    --master_addr "$master_addr" \
    --master_port "$port" \
    main.py -m $mod -b $b -c $c -d $dsz -w $fcw -l $fcl -e $epc \
    --dist \
    --gpus_per_group 3 \
    --group_per_node ${nproc_per_node}

