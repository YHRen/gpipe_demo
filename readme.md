
# Minimum demo on using torchGPipe with DistributedDataParallel

Change the `CUDA_VISIBLE_DEVICES` if needed. 
`bash launch_dist.sh`

`16G` GPU memory is recommanded. If not available, change batch size `b`, model depth `fcl` if needed or model width `fcw` to reduce the model size.

The model is splitted into three GPUs.


## some notes on running IBM's machine using IBM's LSF
To learn more about [LSF](https://www.ibm.com/products/hpc-workload-management)


### on Ascent HPC

```
bsub -P <account> -nnodes 1 -W 60 -alloc_flags gpumps -Is /bin/bash
module load ibm-wml-ce nsight-systems
jsrun -n1 -c16 -g3 -a1 python main.py
jsrun -n1 -c16 -g3 -a1 nsys profile -t nvtx,cuda -s cpu python main.py

jsrun -n1 -c16 -g3 -a1 bash run_script.sh
jsrun -n1 -c16 -g3 -a1 nsys profile -t nvtx,cuda -s cpu bash run_script.sh

```

###
To run eight tasks on four nodes with each task having three GPUs and located
as close as possible to the assigned GPU, run:
```
jsrun --np 8 --nrs 8 --rs_per_host 2 --gpu_per_rs 3 --latency_priority cpu-gpu 
```
