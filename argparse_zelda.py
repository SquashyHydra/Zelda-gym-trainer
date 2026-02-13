from argparse import ArgumentParser
from uuid import uuid4
from sys import argv, path
from os import makedirs
from help_message import help_msg

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")

def get_args(usage_string=None, ep_length=None, sess_path=''):
    #Self-explanatory, gets the arguments given a few arguments that change depending on the file
    if ep_length is None:
        ep_length = 4096
    description='Argument parser for env_config',
    usage=f'python {usage_string} [--headless HEADLESS] [--save_final_state SAVE_FINAL_STATE] ...' #usage different depending on the file
    parser = ArgumentParser(description=description, usage=usage)

    if ('-h' in argv or '--help' in argv):
        print(help_msg)
        exit(0)

    parser.add_argument('--headless', type=str, default="True", help='Whether to run the environment in headless mode')
    parser.add_argument('-save-final-state', '--save_final_state', type=str_to_bool, default=True, help='Whether to save the final state of the environment')
    parser.add_argument('-stop-early', '--early_stop', type=str_to_bool, default=False, help='Whether to stop the environment early')
    parser.add_argument('-action-freq', '--action_freq', type=int, default=8, help='Frequency of actions')
    parser.add_argument('-start-state', '--init_state', type=str, default=f'{path[0]}\\States\\has_shield.state', help='Initial state of the environment')
    parser.add_argument('-max-steps', '--max_steps', type=int, default=ep_length, help='Maximum number of steps in the environment')
    parser.add_argument('-pr', '--print_rewards', type=str_to_bool, default=False, help='Whether to print rewards')
    parser.add_argument('-save-vid', '--save_video', type=str_to_bool, default=False, help='Whether to save a video of the environment')
    parser.add_argument('-save-fast-vid', '--fast_video', type=str_to_bool, default=False, help='Whether to save a fast video of the environment')
    parser.add_argument('-sess-path', '--session_path', type=str, default=sess_path, help='Path to the session')
    parser.add_argument('-cartridge', '--gb_path', type=str, default=f'{path[0]}\\Rom\\Zelda.gb', help='Path to the gameboy ROM')
    parser.add_argument('-d', '--debug', type=str_to_bool, default=False, help='Whether to run the environment in debug mode')
    parser.add_argument('-sim-dist', '--sim_frame_dist', type=float, default=50_000_000.0, help='Simulation frame distance')
    parser.add_argument('-screen-explore', '--use_screen_explore', type=str_to_bool, default=True, help='Use KNN-Index to calculate image difference')
    parser.add_argument('-rs', '--reward_scale',type=int, default=4, help='Multiplication of the earned reward by the reward scale')
    parser.add_argument('-ew', '--explore_weight', type=int, default=3, help='Multiplication of the earned exploration reward by the explore weight')
    parser.add_argument('-pn', '--process_num', type=int, default=12, help='Number of process to run')
    parser.add_argument('-es', '--emulation_speed', type=int, default=6, help='Speed of which the emulator runs in headless')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Minibatch Size')
    parser.add_argument('-is', '--iteration_steps', type=int, default=4096, help='PPO rollout steps per training iteration (n_steps)')
    parser.add_argument('-tt', '--train_timesteps', type=int, default=0, help='Total timesteps to train for (0 uses one rollout: n_steps * num_envs)')
    parser.add_argument('-ti', '--train_iterations', type=int, default=0, help='Number of PPO rollout iterations to train for (0 disables and uses train_timesteps/default)')
    parser.add_argument('-checkpoint', '--checkpoint', type=str, default='', help='Checkpoint of previous training session')
    parser.add_argument('--curriculum_enabled', type=str_to_bool, default=True, help='Enable curriculum reset states (beach -> village -> dungeon)')
    parser.add_argument('--curriculum_beach_state', type=str, default=f'{path[0]}\\States\\text skip.state', help='Initial curriculum state path for beach phase')
    parser.add_argument('--curriculum_village_state', type=str, default=f'{path[0]}\\States\\has_shield.state', help='Initial curriculum state path for village phase')
    parser.add_argument('--curriculum_dungeon_state', type=str, default=f'{path[0]}\\States\\has_shield.state', help='Initial curriculum state path for dungeon-entrance phase')
    parser.add_argument('--curriculum_beach_episodes', type=int, default=50, help='Number of episodes to keep beach curriculum phase')
    parser.add_argument('--curriculum_village_episodes', type=int, default=150, help='Episode index to transition from village to dungeon-entrance phase')
    parser.add_argument('--explore_reward_scale', type=float, default=0.35, help='Scale factor for exploration reward contribution')
    parser.add_argument('--action_penalty_scale', type=float, default=0.25, help='Scale factor for invalid action penalties')
    args, unknown_args= parser.parse_known_args()

    if args.session_path == '':
        args.session_path = f'Sessions\\session_{str(uuid4())[:8]}'
        makedirs(args.session_path, exist_ok=True)

    return args

def change_env(env_config, args):
    #Changes the environment based on the arguments given a env_config dictionary and args
    return {
        **env_config,
        'headless': args.headless,
        'save_final_state': args.save_final_state,
        'early_stop': args.early_stop,
        'action_freq': args.action_freq,
        'init_state': args.init_state,
        'max_steps': args.max_steps,
        'print_rewards': args.print_rewards,
        'save_video': args.save_video,
        'fast_video': args.fast_video,
        'session_path': args.session_path, 
        'gb_path': args.gb_path,
        'debug': args.debug,
        'sim_frame_dist': args.sim_frame_dist,
        'use_screen_explore': args.use_screen_explore,
        'reward_scale': args.reward_scale,
        'explore_weight': args.explore_weight,
        'process_num': args.process_num,
        'emulation_speed': args.emulation_speed,
        'batch_size': args.batch_size,
        'iteration_steps': args.iteration_steps,
        'train_timesteps': args.train_timesteps,
        'train_iterations': args.train_iterations,
        'checkpoint': args.checkpoint,
        'curriculum_enabled': args.curriculum_enabled,
        'curriculum_beach_state': args.curriculum_beach_state,
        'curriculum_village_state': args.curriculum_village_state,
        'curriculum_dungeon_state': args.curriculum_dungeon_state,
        'curriculum_beach_episodes': args.curriculum_beach_episodes,
        'curriculum_village_episodes': args.curriculum_village_episodes,
        'explore_reward_scale': args.explore_reward_scale,
        'action_penalty_scale': args.action_penalty_scale
    }