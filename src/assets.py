import os
from src import _PROJECT_ROOT_
from pathlib import Path

def define_experiment(root: str | Path = os.path.join(_PROJECT_ROOT_, 'runs'), experiment_name: str | Path = None):
    os.makedirs(root, exist_ok=True)
    if not experiment_name:
        experiment_id = str(len(os.listdir(root)))
        experiment_name = f"exp_{experiment_id}"
    experiment_dir = os.path.join(root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

