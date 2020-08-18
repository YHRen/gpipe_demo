## Ascent

bsub -P GEN141 -nnodes 1 -W 60 -alloc_flags gpumps -Is /bin/bash
module load ibm-wml-ce nsight-systems
jsrun -n1 -c16 -g3 -a1 python main.py
jsrun -n1 -c16 -g3 -a1 nsys profile -t nvtx,cuda -s cpu python main.py
