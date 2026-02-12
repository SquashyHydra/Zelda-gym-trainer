from argparse import ArgumentParser
from uuid import uuid4
from sys import argv, path
from os import makedirs
from help_message import help_msg

def get_args(usage_string=None, ep_length=None, sess_path=None):
    #Self-explanatory, gets the arguments given a few arguments that change depending on the file
    if sess_path == None:
        sess_path = f'Sessions\\session_{str(uuid4())[:8]}'
        makedirs(sess_path, exist_ok=True)
    if ep_length == None:
        ep_length = 2048 * 1
    description='Argument parser for env_config',
    usage=f'python {usage_string} [--headless HEADLESS] [--save_final_state SAVE_FINAL_STATE] ...' #usage different depending on the file
    parser = ArgumentParser(description=description, usage=usage)

    if ('-h' in argv or '--help' in argv):
        print(help_msg)
        exit(0)

    parser.add_argument('--headless', type=str, default="True", help='Whether to run the environment in headless mode')
    parser.add_argument('-save-final-state', '--save_final_state', type=bool, default=True, help='Whether to save the final state of the environment')
    parser.add_argument('-stop-early', '--early_stop', type=bool, default=False, help='Whether to stop the environment early')
    parser.add_argument('-action-freq', '--action_freq', type=int, default=8, help='Frequency of actions')
    parser.add_argument('-start-state', '--init_state', type=str, default=f'{path[0]}\\States\\has_shield.state', help='Initial state of the environment')
    parser.add_argument('-max-steps', '--max_steps', type=int, default=ep_length, help='Maximum number of steps in the environment')
    parser.add_argument('-pr', '--print_rewards', type=bool, default=True, help='Whether to print rewards')
    parser.add_argument('-save-vid', '--save_video', type=bool, default=True, help='Whether to save a video of the environment')
    parser.add_argument('-save-fast-vid', '--fast_video', type=bool, default=False, help='Whether to save a fast video of the environment')
    parser.add_argument('-sess-path', '--session_path', type=str, default=sess_path, help='Path to the session')
    parser.add_argument('-cartridge', '--gb_path', type=str, default=f'{path[0]}\\Rom\\Zelda.gb', help='Path to the gameboy ROM')
    parser.add_argument('-d', '--debug', type=bool, default=False, help='Whether to run the environment in debug mode')
    parser.add_argument('-sim-dist', '--sim_frame_dist', type=float, default=50_000_000.0, help='Simulation frame distance')
    parser.add_argument('-screen-explore', '--use_screen_explore', type=bool, default=True, help='Use KNN-Index to calculate image difference')
    parser.add_argument('-rs', '--reward_scale',type=int, default=4, help='Multiplication of the earned reward by the reward scale')
    parser.add_argument('-extra-buttons', '--extra_buttons', type=bool, default=False, help='Use buttons such as start and select on the gameboy controller')
    parser.add_argument('-ew', '--explore_weight', type=int, default=3, help='Multiplication of the earned exploration reward by the explore weight')
    parser.add_argument('-pn', '--process_num', type=int, default=12, help='Number of process to run')
    parser.add_argument('-es', '--emulation_speed', type=int, default=6, help='Speed of which the emulator runs in headless')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Minibatch Size')
    parser.add_argument('-checkpoint', '--checkpoint', type=str, default='', help='Checkpoint of previous training session')
    args, unknown_args= parser.parse_known_args()
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
        'extra_buttons': args.extra_buttons,
        'explore_weight': args.explore_weight,
        'process_num': args.process_num,
        'emulation_speed': args.emulation_speed,
        'batch_size': args.batch_size,
        'checkpoint': args.checkpoint
    }