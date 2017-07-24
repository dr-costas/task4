#!/usr/bin/env bash

#SBATCH -J "dcase"
#SBATCH -o output/out_%A.txt
#SBATCH -e error/err_%A.txt
#SBATCH --mem-per-cpu=45000
#SBATCH -n 1
#SBATCH -t 6-23:59:00
##SBATCH --gres=gpu:k80:1
#SBATCH --gres=gpu:1
#SBATCH --qos=unkillable
#SBARCH -C "gpu12gb"
#SBATCH -x mila01,kepler4,kepler5
#SBATCH --array=1

export PYTHONPATH=$PYTHONPATH:.
srun python scripts/train_one_hot_single.py \
    attend_to_detect.configs.config_single \
    checkpoints/$SLURM_JOB_ID \
    --visdom --visdom-port 5001 --visdom-server http://eos11 \
    --no-tqdm \
    --job-id $SLURM_JOB_ID

