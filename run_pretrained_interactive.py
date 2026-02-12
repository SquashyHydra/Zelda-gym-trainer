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

if __name__ == "__main__":
    ep_length = 2**23
    s_path = path[0] / Path("Sessions")
    makedirs(s_path, exist_ok=True)
    pt_path = s_path / Path(f"Pretrained")
    makedirs(pt_path, exist_ok=True)
    sess_path = pt_path / Path(f'session_{str(uuid4())[:8]}')
    makedirs(sess_path, exist_ok=True)
    agent_file = sess_path / Path(f"agent_enable.txt")
    if not agent_file.is_file():
        with open(agent_file, 'w') as f:
            f.write("yes")
    
    args = get_args()
    env_config = change_env(env_config_default, args)

    if not cuda.is_available():
        print(f"CUDA: {cuda.is_available()}")
        num_proc = 1
        proc_device = device("cuda" if cuda.is_available() else "cpu")
    else:
        print(f"CUDA: {cuda.is_available()}")
        num_proc = 1
        proc_device = device("cuda" if cuda.is_available() else "cpu")

    env = make_env(0, env_config)()

    print('\nloading checkpoint')
    model = PPO.load(f'{path[0]}\\{env_config['checkpoint']}', env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0}, device=proc_device)
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