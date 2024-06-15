import os
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from collections import defaultdict
# Path to your TensorBoard logs directory
logdir = 'baselines/ae_baselines/from_scratch/run'

# TODO: Adapt the scaling to action repeat for episodic return (logged 2x as often as without)
#

new_min = 0
new_max = 1000000

def rescale_steps(scalars):
    # Rescale x-axis data
    original_steps = np.array([s.step for s in scalars])
    original_min = original_steps.min()
    original_max = original_steps.max()
    rescaled_steps = ((original_steps - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min
    return rescaled_steps

# Function to read and log TensorBoard data to WandB
def log_tensorboard_to_wandb(logdir):
    for subdir in os.listdir(logdir):
        run = wandb.init(project="PPO_Causal", name=subdir, sync_tensorboard=True, group='ae_run_baseline')
        subdir_path = os.path.join(logdir, subdir)
        summary_iterators = [EventAccumulator(os.path.join(subdir_path, dname)).Reload() for dname in os.listdir(subdir_path)]
        scalar_data = {}
        rescaled_scalar_data = {}
        for tag in summary_iterators[0].Tags()['scalars']:
            scalar_data[tag] = summary_iterators[0].Scalars(tag)
        for tag, scalars in scalar_data.items():
            rescaled = rescale_steps(scalars)
            rescaled_scalar_data[tag] = [(s.step, s.value, rescaled_step) for s, rescaled_step in zip(scalars, rescaled)]
        run.define_metric("epoch", hidden=True)
        for tag, rescaled_data in rescaled_scalar_data.items():
            for original_step, value, rescaled_step in rescaled_data:
                run.log({tag: value,
                         'global_step': int(rescaled_step),
                         'original_step': original_step}, step=original_step)
        # Finish the run
        run.finish()

# Log all events from the logdir to WandB
log_tensorboard_to_wandb(logdir)
