import os
import pathlib
causal_walk_seeds = [81, 52, 16, 46, 24]
causal_walk_retrain_seeds = [62, 27, 24, 2, 87]

causal_stand_seeds = [98, 89, 1, 42, 70]
causal_stand_retrain_seeds = [28, 83, 72, 54, 53]

causal_run_seeds = [12, 5, 20, 49, 86]
causal_run_retrain_seeds = [84, 79, 64, 71, 40]

test_seed = 35

file_loc = str(pathlib.Path(__file__).parent.resolve()) + '\ppo_causal.py'

os.system(f"python {file_loc} --seed {causal_walk_seeds[0]} --causal")
os.system(f"python {file_loc} --seed {causal_walk_seeds[1]} --causal")
os.system(f"python {file_loc} --seed {causal_walk_seeds[2]} --causal")
# os.system(f"python {file_loc} --seed {causal_walk_seeds[3]} --causal")
# os.system(f"python {file_loc} --seed {causal_walk_seeds[4]} --causal")

# os.system(f"python {file_loc} --seed {causal_walk_retrain_seeds[0]} --eval_seed {causal_walk_seeds[0]} --causal --eval_representation --hard_rep")
# os.system(f"python {file_loc} --seed {causal_walk_retrain_seeds[1]} --eval_seed {causal_walk_seeds[1]} --causal --eval_representation --hard_rep")
# os.system(f"python {file_loc} --seed {causal_walk_retrain_seeds[2]} --eval_seed {causal_walk_seeds[2]} --causal --eval_representation --hard_rep")
# os.system(f"python {file_loc} --seed {causal_walk_retrain_seeds[3]} --eval_seed {causal_walk_seeds[3]} --causal --eval_representation --hard_rep")

# os.system(f"python {file_loc} --seed {causal_walk_seeds[0]} --causal --eval_representation --eval_seed {causal_walk_seeds[0]}")
