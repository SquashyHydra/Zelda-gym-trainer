help_msg = f"""usage: test.py [-h] [--headless] [--save_final_state] [--early_stop] [--action_freq] [--init_state] [--max_steps]
[--print_rewards] [--save_video] [--fast_video] [--session_path] [--gb_path] [--debug] [--sim_frame_dist]
[--use_screen_explore] [--reward_scale] [--extra_buttons] [--explore_weight] [--process_num]

optional arguments:
  -h, --help            show this help message
  --headless    Set the environment to headless mode or not (default: True)
  -save-final-state, --save_final_state    Save the final state of the simulation (default: False)
  -stop-early, --early_stop    Early stop the episode if the state matches the initial state (default: False)
  -action-freq, --action_freq   Frequency to perform an action (default: 8)
  -start-state, --init_state    Initial state of the environment (default: None)
  -max-steps, --max_steps  Maximum number of steps per episode (default: 2048 * 1)
  -pr, --print_rewards   Print rewards for each step (default: True)
  -save-vid, --save_video    Save video during the simulation (default: False)
  -save-fast-vid, --fast_video    Fast forward through the video while recording (default: False)
  -sess-path, --session_path    Session path for saving files (default: None)
  -cartridge, --gb_path   Path to the game boy rom (default: None)
  -d, --debug   Debug mode (default: False)
  -sim-dist, --sim_frame_dist    Number of simulation frames before taking an action (default: 50000000.0)
  -screen-explore, --use_screen_explore    Use screen exploration instead of the default exploration method (default: False)
  -rs, --reward_scale    Reward scale for earned rewards (default: 4)
  -extra-buttons, --extra_buttons   Use extra buttons such as start and select on the gameboy controller (default: False)
  -ew, --explore_weight    Multiplication of the earned exploration reward by this weight (default: 3)
  -pn, --process_num    Number of processes to run in parallel (default: 12)
  -es, --emulation_speed    Speed of which the emulator runs in headless (default: 6)
  -bs, --batch_size    Minibatch Size (default: 64)
  -checkpoint --checkpoint    Checkpoint of previous training session (example: Sessions\\session_7eaf63ba\\zelda_2383872_steps)  """