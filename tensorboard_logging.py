import os
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
# Path to your TensorBoard logs directory
logdir = 'runs'

# TODO: Rescale the step size (500 for the episodic_return and 2048 for others)
#   Also define groups based on the name before the first underscore

# Function to read and log TensorBoard data to WandB
def log_tensorboard_to_wandb(logdir):
    for subdir in os.listdir(logdir):
        run = wandb.init(project="PPO_Causal", name=subdir, sync_tensorboard=True)
        subdir_path = os.path.join(logdir, subdir)
        summary_iterators = [EventAccumulator(os.path.join(subdir_path, dname)).Reload() for dname in os.listdir(subdir_path)]
        tags = summary_iterators[0].Tags()['scalars']
        run.define_metric("epoch", hidden=True)
        for tag in tags:
            events = summary_iterators[0].Scalars(tag)
            for event in events:
                run.log({tag: event.value}, step=event.step)
        # Finish the run
        run.finish()

# Log all events from the logdir to WandB
log_tensorboard_to_wandb(logdir)
