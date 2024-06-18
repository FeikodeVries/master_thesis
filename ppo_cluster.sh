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
#module load cuda11.1/toolkit
#module load cuDNN/cuda11.1

# module load cuda11.3/toolkit/11.3.1

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate

# Base directory for the experiment
# mkdir $HOME/experiments
cd $HOME/experiments

hidden_dims=(64 128 256 512 1024)
latent_space=(25 64 84 96 128 256 512)
seeds=(45 37 7)
act_fns=('relu', 'tanh', 'silu')
latent=${latent_space[SLURM_ARRAY_TASK_ID]}
c_hid=${hidden_dims[SLURM_ARRAY_TASK_ID]}
act_fn=${act_fns[SLURM_ARRAY_TASK_ID]}
seed=${seeds[SLURM_ARRAY_TASK_ID]}

# Simple trick to create a unique directory for each run of the script
echo $$ $seed $1
mkdir o`echo $$`_dmc_ae_walk_nofreeze_seed$seed
cd o`echo $$`_dmc_ae_walk_nofreeze_seed$seed

# Run the actual experiment.
MUJOCO_GL=egl python -u /home/fvs660/cleanrl/cleanrl/ppo_causal.py --seed $seed --curl --ae_freeze 2 --curl_encoder_update_freq 4
