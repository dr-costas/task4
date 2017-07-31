#!/usr/bin/env bash

#SBATCH -J "dcase"
#SBATCH -o output/out_%A.txt
#SBATCH -e error/err_%A.txt
##SBATCH --mem-per-cpu=45000
#SBATCH -n 1
#SBATCH -t 6-23:59:00
#SBATCH --gres=gpu:1
#SBATCH --qos=unkillable
#SBATCH -C "gpu24gb"
#SBATCH -x mila01,kepler2,kepler3
#SBATCH --nodelist=kepler5
#SBATCH --array=1

export PYTHONPATH=$PYTHONPATH:.
srun python scripts/train_model_new_approach.py \
    attend_to_detect.configs.config_new_model_single_mel_only \
    /Tmp/drososko/checkpoints/$SLURM_JOB_ID \
    --visdom --visdom-port 5001 --visdom-server http://eos11 \
    --no-tqdm \
    --job-id $SLURM_JOB_ID

