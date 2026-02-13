import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings; warnings.filterwarnings("ignore", category=UserWarning, message="Using SDL2 binaries from pysdl2-dll")

from torch import cuda, device
from sys import path
from pathlib import Path
from os.path import exists
from os import cpu_count

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from ZeldaGym.env import ZeldaGymEnv
from tensorboard_callback import TensorboardCallback
from argparse_zelda import get_args, change_env
from enviroment_config import env_config_default
from helper import get_latest_checkpoint, get_checkpoint_step

def make_env(rank, env_conf, seed=0):
    def _init():
        env = ZeldaGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    use_wandb_logging = False
    args = get_args()
    env_config = change_env(env_config_default, args)
    ep_length = int(env_config.get('iteration_steps') or 4096)

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
    env = VecTransposeImage(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.998)

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=env_config['session_path'], name_prefix='zelda', save_vecnormalize=True)

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

    checkpoint_folder = f"{path[0] / Path(env_config['checkpoint']) if env_config['checkpoint'] not in [None, ''] else Path(env_config['session_path'])}"
    checkpoint_path = get_latest_checkpoint(checkpoint_folder)
    checkpoint_step = get_checkpoint_step(checkpoint_path)
    vecnorm_path = None
    if checkpoint_step is not None:
        vecnorm_path = f"{checkpoint_folder}\\zelda_vecnormalize_{checkpoint_step}_steps.pkl"

    if vecnorm_path is not None and exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True

    if checkpoint_path is not None and exists(checkpoint_path):
        print('loading checkpoint')
        model = PPO.load(checkpoint_path, env=env, device=proc_device)
        print('checkpoint loaded')
        model.n_steps = ep_length
        model.n_envs = num_proc
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_proc
        model.gamma = 0.998
        model.tensorboard_log = env_config['session_path']
        model.device = proc_device
        model.rollout_buffer.reset()
    else:
        policy_kwargs = dict(normalize_images=True)
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=env_config['batch_size'], n_epochs=3, gamma=0.998, tensorboard_log=env_config['session_path'], device=proc_device, policy_kwargs=policy_kwargs)

    if env_config.get('train_iterations') and env_config['train_iterations'] > 0:
        total_timesteps = int(env_config['train_iterations']) * ep_length * num_proc
    elif env_config.get('train_timesteps') and env_config['train_timesteps'] > 0:
        total_timesteps = int(env_config['train_timesteps'])
    else:
        total_timesteps = ep_length * num_proc
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()