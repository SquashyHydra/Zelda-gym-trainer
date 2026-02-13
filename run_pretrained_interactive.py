import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings; warnings.filterwarnings("ignore", category=UserWarning, message="Using SDL2 binaries from pysdl2-dll")

from torch import cuda, device
from sys import path
from pathlib import Path
from os import makedirs

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize

from ZeldaGym.env import ZeldaGymEnv
from argparse_zelda import get_args, change_env
from enviroment_config import env_config_default
from helper import get_latest_checkpoint, get_checkpoint_step

def make_env(rank, env_conf, seed=0):
    def _init():
        env = ZeldaGymEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    ep_length = 2**23
    args = get_args()
    env_config = change_env(env_config_default, args)
    sess_path = Path(env_config['session_path'])

    pt_path = Path(path[0]) / Path("Sessions") / Path("Pretrained")
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

    base_env = DummyVecEnv([make_env(0, env_config)])
    env = VecTransposeImage(base_env)

    checkpoint_folder = f"{path[0] / Path(env_config['checkpoint']) if env_config['checkpoint'] not in [None, ''] else Path(env_config['session_path'])}"
    checkpoint_path = get_latest_checkpoint(checkpoint_folder) if env_config['checkpoint'] else None
    checkpoint_step = get_checkpoint_step(checkpoint_path)

    if checkpoint_path is None:
        print(f"No checkpoint found at {checkpoint_folder}")
        exit(1)

    if checkpoint_step is not None:
        vecnorm_path = f"{checkpoint_folder}\\zelda_vecnormalize_{checkpoint_step}_steps.pkl"
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

    print('\nloading checkpoint')
    model = PPO.load(checkpoint_path, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0}, device=proc_device)
    print('\ncheckpoint loaded')

    #keyboard.on_press_key("M", toggle_agent)
    obs = env.reset()
    action = 6 # default action
    while True:
        try:
            with open(agent_file, 'r') as f:
                agent_enabled = f.read().strip().lower() == "yes"
        except OSError:
            agent_enabled = False

        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)
        env.env_method("render")

        if done[0]:
            break
    env.close()