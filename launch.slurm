#!/bin/bash
#SBATCH --account=pmonteag
#SBATCH --job-name=kl_spinquant  # job name for reference
#SBATCH --nodes=1                   # job on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=32          # job name for referance
#SBATCH --nice=100000               # Very low priority job
#SBATCH --time=4:00:00             # time limit of the job in hh:mi:ss
#SBATCH --exclude=radha0            # job on a single node
#SBATCH --output=test_%A_%a.log     # Standard output and error log

echo $HOSTNAME
free -h
rocm-smi
source /home/pmonteag/.local/miniforge3/etc/profile.d/conda.sh
conda activate brv_llm_rocm
cd /home/pmonteag/brevitas

HF_HUB_CACHE=/home/pmonteag python /home/pmonteag/brevitas/src/brevitas_examples/llm/benchmark/llm_benchmark.py --config /home/pmonteag/brevitas/src/brevitas_examples/common/benchmark/rotation_benchmark_llama_adam.yaml --results-folder /home/pmonteag/brevitas/results_llama  --gpus="0,1,2,3"