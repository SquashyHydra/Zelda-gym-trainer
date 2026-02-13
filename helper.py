import os
from os.path import exists

def get_latest_checkpoint(checkpoint_folder):
    if not checkpoint_folder or not exists(checkpoint_folder):
        return None
    checkpoint_path = None
    max_iterations = 0
    for file in os.listdir(checkpoint_folder):
        if file.startswith("zelda_") and file.endswith("_steps.zip"):
            total_iterations = file.replace("zelda_", "").replace("_steps.zip", "")
            if total_iterations and max_iterations < int(total_iterations):
                max_iterations = int(total_iterations)
                checkpoint_path = os.path.join(checkpoint_folder, f"zelda_{max_iterations}_steps.zip")
    return checkpoint_path

def get_checkpoint_step(checkpoint_path):
    if not checkpoint_path:
        return None
    file_name = os.path.basename(checkpoint_path)
    if not file_name.startswith("zelda_") or not file_name.endswith("_steps.zip"):
        return None
    step = file_name.replace("zelda_", "").replace("_steps.zip", "")
    return step if step.isdigit() else None