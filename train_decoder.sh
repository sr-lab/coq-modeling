#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=02-11:59:00          # total run time limit (HH:MM:SS)
#SBATCH -o slurm/out/slurm-train-decoder-%j.out
#SBATCH --no-requeue

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=TRAIN        # create a short name for your job
#SBATCH --mem=50GB               # total memory per node


export OPENAI_API_KEY=""
eval $(opam env)
source venv/bin/activate
# python3 scripts/move_data.py confs/train/conf.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch src/tactic_gen/train_decoder.py confs/train/conf.yaml
