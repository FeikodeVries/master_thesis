#!/bin/bash
#SBATCH --job-name=ppo_vae
#SBATCH --time=16:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

## in the list above, the partition name depends on where you are running your job. 
## On DAS5 the default would be `defq` on Lisa the default would be `gpu` or `gpu_shared`
## Typing `sinfo` on the server command line gives a column called PARTITION.  There, one can find the name of a specific node, the state (down, alloc, idle etc), the availability and how long is the time limit . Ask your supervisor before running jobs on queues you do not know.

# Load GPU drivers

## Enable the following two lines for DAS5
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate

# Base directory for the experiment
# mkdir $HOME/experiments
cd $HOME/experiments

ae_walker_walk_seeds=(12 50 83 95 99)
curl_walker_walk_seeds=(2 5 46 60 84)
causal_walker_walk_seeds=(0 0 0 0 0)
seeds=(9 43 55 29)

seed=${seeds[SLURM_ARRAY_TASK_ID]}
ae_eval_seed=${ae_walker_walk_seeds[SLURM_ARRAY_TASK_ID]}
curl_eval_seed=${curl_walker_walk_seeds[SLURM_ARRAY_TASK_ID]}
causal_eval_seed=${causal_walker_walk_seeds[SLURM_ARRAY_TASK_ID]}

# Simple trick to create a unique directory for each run of the script
echo $$ $seed $1
mkdir o`echo $$`_dmc_pixel_stand_$seed
cd o`echo $$`_dmc_pixel_stand_$seed

# Run the actual experiment.
MUJOCO_GL=egl python -u /home/fvs660/cleanrl/cleanrl/ppo_causal.py --seed $seed --env_id "dm_control/walker-stand-v0" --pixel_baseline
