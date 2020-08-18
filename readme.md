## Ascent

bsub -P GEN141 -nnodes 1 -W 60 -alloc_flags gpumps -Is /bin/bash
module load ibm-wml-ce nsight-systems
jsrun -n1 -c16 -g3 -a1 python main.py
jsrun -n1 -c16 -g3 -a1 nsys profile -t nvtx,cuda -s cpu python main.py

jsrun -n1 -c16 -g3 -a1 bash run_script.sh
jsrun -n1 -c16 -g3 -a1 nsys profile -t nvtx,cuda -s cpu bash run_script.sh


# run cnn
jsrun -n1 -c16 -g3 -a1 bash run_script.sh cnn

# PyProf

# run dist
jsrun -n1 -c16 -g3 -a1

# 
To run eight tasks on four nodes with each task having three GPUs and located
as close as possible to the assigned GPU, run:
```
jsrun --np 8 --nrs 8 --rs_per_host 2 --gpu_per_rs 3 --latency_priority cpu-gpu 
```
