import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings; warnings.filterwarnings("ignore", category=UserWarning, message="Using SDL2 binaries from pysdl2-dll")

from torch import cuda, device
from sys import path
from pathlib import Path
from os import makedirs
from uuid import uuid4

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from ZeldaGym.env import ZeldaGymEnv
from argparse_zelda import get_args, change_env
from enviroment_config import env_config_default

def make_env(rank, env_conf, seed=0):
    def _init():
        env = ZeldaGymEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

def get_latest_checkpoint(checkpoint_folder):
    checkpoint_path = None
    max_iterations = 0
    for file in os.listdir(checkpoint_folder):
        if file.endswith(".zip"):
            if file.startswith("zelda"):
                total_iterations = file.replace("zelda_", "").replace("_steps.zip", "")
            if max_iterations < int(total_iterations):
                max_iterations = int(total_iterations)
            checkpoint_path = f"{checkpoint_folder}\\zelda_{max_iterations}_steps.zip"
    return checkpoint_path

if __name__ == "__main__":
    ep_length = 2**23
    args = get_args()
    env_config = change_env(env_config_default, args)
    sess_path = env_config['session_path']

    pt_path = path[0] / Path("Sessions") / Path(f"Pretrained")
    makedirs(pt_path, exist_ok=True)
    agent_file = sess_path / Path(f"agent_enable.txt")
    if not agent_file.is_file():
        with open(agent_file, 'w') as f:
            f.write("yes")

    if not cuda.is_available():
        print(f"CUDA: {cuda.is_available()}")
        num_proc = 1
        proc_device = device("cuda" if cuda.is_available() else "cpu")
    else:
        print(f"CUDA: {cuda.is_available()}")
        num_proc = 1
        proc_device = device("cuda" if cuda.is_available() else "cpu")

    env = make_env(0, env_config)()

    checkpoint_folder = f"{path[0]}\\{env_config['checkpoint']}"
    checkpoint_path = get_latest_checkpoint(checkpoint_folder)

    if checkpoint_path is None:
        print(f"No checkpoint found at {checkpoint_folder}")
        exit(1)

    print('\nloading checkpoint')
    model = PPO.load(checkpoint_path, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0}, device=proc_device)
    print('\ncheckpoint loaded')

    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    action = 6 # default action
    while True:
        try:
            with open(f"{sess_path}/agent_enable.txt", 'r') as f:
                agent_enabled = f.read()

            if agent_enabled == "yes":
                agent_enabled = True
        except Exception as e:
            agent_enabled = False

        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if truncated:
            break
    env.close()