import os
import warnings; warnings.filterwarnings("ignore", category=UserWarning, message="Using SDL2 binaries from pysdl2-dll")

from torch import cuda, device
from sys import path
from pathlib import Path
from os.path import exists
from os import makedirs, cpu_count
from uuid import uuid4

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from ZeldaGym.env import ZeldaGymEnv
from tensorboard_callback import TensorboardCallback
from argparse_zelda import get_args, change_env
from enviroment_config import env_config_default

def make_env(rank, env_conf, seed=0):
    def _init():
        env = ZeldaGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    use_wandb_logging = False
    ep_length = 2048 * 1
    sess_id = str(uuid4())[:8]
    sess_path = path[0] / Path(f"Sessions/session_{sess_id}")
    makedirs(sess_path, exist_ok=True)

    args = get_args()
    env_config = change_env(env_config_default, args)

    if not cuda.is_available():
        print(f"CUDA: {cuda.is_available()}")
        num_proc = env_config['process_num']
        if num_proc > cpu_count():
            num_proc = cpu_count()
        proc_device = device("cuda" if cuda.is_available() else "cpu")
    else:
        print(f"CUDA: {cuda.is_available()}")
        num_proc = env_config['process_num']
        proc_device = device("cuda" if cuda.is_available() else "cpu")

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_proc)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=env_config['session_path'], name_prefix='zelda')

    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="zelda-train",
            id=env_config['session_path'][-8:],
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    if exists(f"{path[0]}\\{env_config['checkpoint']}" + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(f'{path[0]}\\{env_config['checkpoint']}', env=env, device=proc_device)
        print('\ncheckpoint loaded')
        model.n_steps = ep_length
        model.n_envs = num_proc
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_proc
        model.n_envs = 3
        model.gamma = 0.998
        model.tensorboard_log = env_config['checkpoint'][:25]
        model. device = device
        model.rollout_buffer.reset()
    else:
        model = PPO('MlpPolicy', env, verbose=1, n_steps=ep_length, batch_size=env_config['batch_size'], n_epochs=3, gamma=0.998, tensorboard_log=env_config['session_path'], device=proc_device)

    model.learn(total_timesteps=(ep_length)*num_proc, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()