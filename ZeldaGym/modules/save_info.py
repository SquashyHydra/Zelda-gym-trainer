from os import makedirs
from json import dump
from pandas import DataFrame
from pathlib import Path
from ZeldaGym.modules.media import image_save
from warnings import catch_warnings, simplefilter

def append_agent_statistics(s_path, instance_id, all_runs, agent_stats):
    agent_dir = s_path / Path('Agent_Stats')
    makedirs(agent_dir, exist_ok=True)
    at_dir = s_path / Path(f'Agent_Stats/{instance_id}')
    makedirs(at_dir, exist_ok=True)
    with open(agent_dir / Path(f'all_runs_{instance_id}.json'), 'w') as f:
        dump(all_runs, f, indent=4)
    with catch_warnings():
        simplefilter("ignore")
        DataFrame(agent_stats).to_csv(at_dir / Path(f'agent_stats_{instance_id}.csv.zip'), compression="zip", mode='w')

def append_final_stats(save_final_state, s_path, instance_id, reset_count, total_reward, render):
    if save_final_state:
        fs_dir = s_path / Path('Images/final_states')
        makedirs(fs_dir, exist_ok=True)
        inst_dir = fs_dir / Path(f"id_{instance_id}")
        makedirs(inst_dir, exist_ok=True)
        image_save(inst_dir / Path(f'frame_{reset_count}_r{total_reward:.4f}_full.jpg'), render)

def append_curr_frame(s_path, reset_count, instance_id, render):
    image_dir = s_path / Path('Images')
    makedirs(image_dir, exist_ok=True)
    cf_dir = s_path / Path('Images/current_frame')
    makedirs(cf_dir, exist_ok=True)
    image_save(cf_dir / Path(f'curframe_{reset_count}_{instance_id}.jpg'), render)

def append_print_info(print_rewards, instance_id, step_count, progress_reward, total_reward, vec_dim, obs_dim):
    if print_rewards:
        prog_string = f'{"":44} id: {instance_id} | step: {step_count} | '
        for key, val in progress_reward.items():
            if key == "explore" or key == "action" or key == "Dist Diff":
                prog_string += f'\n{"":44} '
            if isinstance(val, str):
                prog_string += f'{key}: {val} | '
            else:
                prog_string += f'{key}: {val:5.2f} | '

        prog_string += f'\n{"":44} Total_Rewards: {total_reward:5.2f} | vec_dim: {vec_dim} | obs_vector_dim: {obs_dim}'
        print(f'{prog_string}\033[F\033[F\033[F\033[F', end='\r', flush=True)