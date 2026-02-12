import os, uuid, hnswlib

import numpy as np
import mediapy as media

#from cv2 import resize
from skimage.transform import resize
from pyboy import PyBoy
from gymnasium import Env, spaces
from pathlib import Path
from einops import rearrange

from ZeldaGym.modules.actions import valid_actions, release_arrows, release_buttons
from ZeldaGym.modules.mem_addresses import (world_map_status_addr, x_pos_addr, y_pos_addr, destination_data_addr, current_health_addr,
                                            items_held_addr, inventory_add, item_levels_addr, have_items_addr, number_items_addr,
                                            max_items_addr, cur_loaded_map, dungeon_flags_addr)
from ZeldaGym.modules.actions import get_action_name

class ZeldaGymEnv(Env):

    def __init__(
        self, config=None):

        self.debug = config['debug']
        self.s_path = Path(config['session_path'])
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.headless = config['headless']
        self.emulation_speed = config['emulation_speed']
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.similar_frame_dist = config['sim_frame_dist']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']

        # Input Map
        self.valid_actions = valid_actions
        self.release_arrow = release_arrows
        self.release_button = release_buttons
        
        # Initialize Variables
        self.vec_dim = 11520 # if ever changed check obs_flat for the new vector dimention
        self.obs_dim = 0 # this is just to display (for now who knows)
        self.num_elements = 20000 # max
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.reset_count = 0
        self.all_runs = []
        self.previous_item_held = (0, 0)
        self.current_item_held = (0, 0)
        self.dist_diff = 0
        self.distance = 0
        self.labels = None

        # Create Shapes
        self.output_shape = (36, 40, 8)
        self.replay_shape = (144, 160, 4)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height), self.output_shape[1],self.output_shape[2])

        # These are for gymnasuim find put what they do https://gymnasium.farama.org/api/spaces/# (https://github.com/Baekalfen/PyBoy/wiki/Migrating-from-v1.x.x-to-v2.0.0)
        self.action_space = spaces.Discrete(len(self.valid_actions)) # (https://gymnasium.farama.org/api/spaces/fundamental/#discrete) https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8) # https://gymnasium.farama.org/api/spaces/fundamental/#box

        # Setup pyboy https://docs.pyboy.dk
        if config['headless'].lower() == "true":
            self.pyboy = PyBoy(config['gb_path'],window="null",) 
        else:
            if config['headless'] == 'OpenGL':
                self.pyboy = PyBoy(config['gb_path'],window="OpenGL",)
            else:
                print('pyboy config window set')
                self.pyboy = PyBoy(config['gb_path'],window="SDL2",)

        self.screen = self.pyboy.screen
        self.pyboy.set_emulation_speed(0 if config['headless'].lower() == "true" else config['emulation_speed'])
        self.reset()



    # MARK: Reset
    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem()

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, self.output_shape[2]), dtype=np.uint8)

        self.recent_frames = np.zeros((self.frame_stacks, self.output_shape[0],self.output_shape[1], self.output_shape[2]),dtype=np.uint8)
        
        self.previous_item_held = (self.read_m(0xDB00), self.read_m(0xDB01)) # holding a shield (4, 0)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('Saved_Video')
            os.makedirs(base_dir, exist_ok=True)

            model_dir = base_dir / Path(f'Model_Video')
            full_dir = model_dir / Path(f'id_{self.instance_id}')
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(full_dir, exist_ok=True)
            full_name = full_dir / Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.model_frame_writer = media.VideoWriter(full_name, (self.output_full[:2]), fps=60, codec='h264')
            self.model_frame_writer.__enter__()

            replay_dir = base_dir / Path(f'Replay_Video')
            full_dir = replay_dir / Path(f'id_{self.instance_id}')
            os.makedirs(replay_dir, exist_ok=True)
            os.makedirs(full_dir, exist_ok=True)
            full_name = full_dir / Path(f'replay_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.replay_frame_writer = media.VideoWriter(full_name, (self.replay_shape[:2]), fps=60, codec='h264')
            self.replay_frame_writer.__enter__()

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = self.read_hp(current_health_addr)
        self.total_healing_rew = 0
        self.total_held_item_rew = 0
        self.total_tile_explore = 0
        self.total_coords_reward = 0
        self.total_action_reward = 0
        self.last_seen_tile_coords = {}
        self.ai_action = 0
        self.died_count = 0
        self.step_count = 0
        self.current_level = self.get_levels_sum()
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self.render(), {}
    


    # MARK: Render
    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixel_render = self.screen.ndarray # (144, 160, 4)
        if reduce_res:
            game_pixel_render = (255*resize(game_pixel_render, self.output_shape)).astype(np.int8)
            if update_mem:
                self.recent_frames[0] = game_pixel_render
            if add_memory:
                pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], self.output_shape[2]), dtype=np.uint8)
                game_pixel_render = np.concatenate((self.create_exploration_memory(), pad, self.create_recent_memory(), pad, rearrange(self.recent_frames, "f h w c -> (f h) w c")), axis=0)
        return game_pixel_render
    


    # MARK: Step
    def step(self, action):
        # Get Values from agent
        self.current_level = self.get_levels_sum()
        self.ai_action = action
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)


        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)
        self.obs_dim = obs_flat
        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords()

        self.update_heal_reward() # needs to be fixed don't understand how healing works in the game yet

        #Add new rewards here
        self.update_held_item()
        self.update_seen_tiles()
        self.get_seen_coord()
        self.get_action_reward()

        new_reward, new_prog = self.update_reward()

        self.last_health = self.read_m(current_health_addr)

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, self.output_shape[2])
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}



    # MARK: Actions
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == self.act_freq-1:
                if int(action) < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if int(action) >= 4 and action <= 5:   
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])
                if int(action) > 5:
                    # release Start Button
                    self.pyboy.send_input(self.release_button[action - 5])

            if self.save_video and not self.fast_video:
                self.add_video_frame()

            if i == self.act_freq-1:
                self.pyboy._tick(True)

            self.pyboy.tick()

        if self.save_video and self.fast_video:
            self.add_video_frame()



    # MARK: Agent Stats
    def append_agent_stats(self, action):
        x_pos = self.read_m(x_pos_addr)
        y_pos = self.read_m(y_pos_addr)
        map_p = self.seen_tile_coords
        map_w = self.seen_tiles
        dungeon_flags = [self.read_m(a) for a in dungeon_flags_addr]
        hands = [self.read_m(a) for a in items_held_addr] # What is in Hands
        Inventory = [self.read_m(a) for a in inventory_add] # What is in Inventory
        item_levels = [self.read_m(a) for a in item_levels_addr] 
        item_have = [self.read_m(a) for a in have_items_addr] # return 0 if no item or 1 if is item
        item_numbers = [self.read_m(a) for a in number_items_addr]
        item_max = [self.read_m(a) for a in max_items_addr]
        explore_count = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        self.agent_stats.append({
            'step': self.step_count,
            'last_action': action,
            'x-position': x_pos,
            'y-position': y_pos,
            'items_hand': hands, 'items_inventory': Inventory,
            'item_levels': item_levels,
            'items_have': item_have,
            'items_numbers': item_numbers,
            'items_max': item_max,
            'health': self.read_m(current_health_addr),
            'explore_count': explore_count,
            'deaths': self.died_count,
            'healed': self.total_healing_rew,
            'dungeon_flags': dungeon_flags,
            'world': str(map_w),
            'Seen_tile_coords': str(map_p),
        })



    # MARK: Initialize Map Memory    
    def init_map_mem(self):
        self.seen_coords = {}
        self.seen_tiles = set()
        self.seen_tile_coords = {}
        self.current_loaded_map = set()

    #Update Coords
    def update_seen_coords(self):
        x_pos = self.read_m(x_pos_addr)
        y_pos = self.read_m(y_pos_addr)
        map_n = [self.read_m(a) for a in destination_data_addr]
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}
            
        self.seen_coords[coord_string] = self.step_count



    # MARK: Create AI Memory
    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        from ZeldaGym.modules.rewards import make_reward_channel
        level, health, dead, explore, tile_explore, coord_explore, action, held_item = self.group_rewards() # add the rest when done
        full_memory = np.stack((
            make_reward_channel(level, self.col_steps, h, w,),
            make_reward_channel(health, self.col_steps, h, w),
            make_reward_channel(dead, self.col_steps, h, w),
            make_reward_channel(explore, self.col_steps, h, w),
            make_reward_channel(tile_explore, self.col_steps, h, w),
            make_reward_channel(coord_explore, self.col_steps, h, w),
            make_reward_channel(action, self.col_steps, h, w),
            make_reward_channel(held_item, self.col_steps, h, w),
        ), axis=-1)
        return full_memory
    
    def create_recent_memory(self):
        return rearrange(
            self.recent_memory,
            '(w h) c -> h w c',
            h=self.memory_height)



    # MARK: KNN (K-Nearest-Neighbor)
    def init_knn(self):
        self.seen_tiles = set()
        self.seen_tile_coords = {}
        self.current_loaded_map = ()
        #Declaring Index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # https://github.com/nmslib/hnswlib
        #initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)

    def get_knn_reward(self):
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01

        #calculate exploration reward based on current exploration state
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew

        return base + post

    def update_frame_knn_index(self, frame_vec):

        if self.get_levels_sum() >= 9 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            #if index is empty add current frame
            self.knn_index.add_items(frame_vec, np.array([self.knn_index.get_current_count()]))
        else:
            # check for nearest frame and add if current
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            self.labels = labels
            self.distance = distances[0]
            if distances[0] != 0:
                self.dist_diff = self.similar_frame_dist / self.distance * 100
            map_w = tuple([self.read_m(a) for a in world_map_status_addr])
            if map_w not in self.seen_tiles and distances[0]  != 0:    #if distances[0] > self.similar_frame_dist:
                self.knn_index.add_items(frame_vec, np.array([self.knn_index.get_current_count()]))



    # MARK: Rewards
    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()

        new_total = sum(self.progress_reward.values()) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward

        if new_step < 0 and self.read_m(current_health_addr) > 0:
            self.save_screenshot('neg_reward')

        self.total_reward = new_total
        return new_step, tuple(np.subtract(new_prog, old_prog))

    def get_game_state_reward(self):
        state_scores = {
            'level': self.get_levels_reward(),
            'heal': self.total_healing_rew,
            'dead': self.reward_scale*-0.4*self.died_count,
            'explore': self.reward_scale * (self.explore_weight * self.get_knn_reward()), # will have to do for now till can sort out the vector dimestion so i can add more rewards [((self.reward_scale * (self.explore_weight * self.get_knn_reward())) + (self.reward_scale * self.total_tile_explore) + (self.reward_scale * self.total_coords_reward)) * 10]
            'tile_explore': (self.reward_scale * self.total_tile_explore),
            'coord_explore': (self.reward_scale * self.total_coords_reward),
            'action': (self.reward_scale * self.total_action_reward),
            'held_item': self.total_held_item_rew,
        }
        return state_scores
    
    def group_rewards(self):
        prog = self.progress_reward
        return (prog['level'], 
                prog['heal'],
                prog['dead'],
                prog['explore'],
                prog['tile_explore'],
                prog['coord_explore'],
                prog['action'],
                prog['held_item'])

    def update_heal_reward(self):
        cur_health = self.read_m(current_health_addr)
        from ZeldaGym.modules.rewards import calc_heal_rew
        self.total_healing_rew, self.died_count = calc_heal_rew(cur_health,self.last_health, self.total_healing_rew, self.died_count, self.reward_scale)
    def update_held_item(self):
        self.current_item_held = (self.read_m(0xDB00), self.read_m(0xDB01))
        from ZeldaGym.modules.rewards import calc_held_item_rew
        self.previous_item_held, self.total_held_item_rew = calc_held_item_rew(self.current_item_held, self.previous_item_held, self.reward_scale, self.total_held_item_rew)
    def update_seen_tiles(self):
        map_w = tuple([self.read_m(a) for a in world_map_status_addr])
        from ZeldaGym.modules.rewards import calc_tile_rew
        self.total_tile_explore, self.seen_tiles = calc_tile_rew(map_w, self.seen_tiles, self.explore_weight, self.total_tile_explore)
    def get_levels_reward(self):
        from ZeldaGym.modules.rewards import calc_level_rew
        return calc_level_rew(self.reward_scale, self.get_levels_sum(), self.max_level_rew, self.current_level)
    def get_seen_coord(self):
        x_pos = self.read_m(x_pos_addr); y_pos = self.read_m(y_pos_addr)
        map_loaded = str(tuple([self.read_m(a) for a in world_map_status_addr]))
        if self.seen_tile_coords:
            self.last_seen_tile_coords = self.seen_tile_coords
        from ZeldaGym.modules.rewards import calc_tile_coords_rew
        self.total_coords_reward, self.seen_tile_coords = calc_tile_coords_rew(map_loaded, (x_pos, y_pos), self.seen_tile_coords, self.total_coords_reward, self.explore_weight)
    def get_action_reward(self):
        from ZeldaGym.modules.rewards import calc_action_rew
        self.total_action_reward = calc_action_rew(self.seen_tile_coords, self.ai_action, self.total_action_reward, self.last_seen_tile_coords, self.reward_scale)

    # MARK: Save and Print Information, it's in the name
    def save_and_print_info(self, done, obs_memory):
        prog_rew = self.progress_reward
        prog_rew['Action'] = int(self.ai_action)
        prog_rew['Dist Diff'] = f'{float(self.dist_diff):7.2f}%'
        prog_rew['Distances[0]'] = float(self.distance)
        prog_rew['Sim_Dist_Diff'] = self.similar_frame_dist
        from ZeldaGym.modules.save_info import append_print_info
        append_print_info(self.print_rewards, self.instance_id, self.step_count, prog_rew, self.total_reward, self.vec_dim, self.obs_dim.shape[0])

        if self.step_count % 50 == 0:
            from ZeldaGym.modules.save_info import append_curr_frame
            append_curr_frame(self.s_path, self.reset_count, self.instance_id, self.render(reduce_res=False))

        if self.print_rewards and done:
            from ZeldaGym.modules.save_info import append_final_stats
            append_final_stats(self.save_final_state, self.s_path, self.instance_id, self.reset_count, self.total_reward, self.render(reduce_res=False))

        if self.save_video and done:
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            from ZeldaGym.modules.save_info import append_agent_statistics
            append_agent_statistics(self.s_path, self.instance_id, self.all_runs, self.agent_stats)



    # MARK: Media
    def save_screenshot(self, name):
        from ZeldaGym.modules.media import screenshot
        screenshot(self.s_path, self.instance_id, self.reset_count, self.total_reward, name, self.render(reduce_res=False))

    def add_video_frame(self):
        model_frame = self.render(reduce_res=True, update_mem=False)
        model_frame = model_frame[:, :, :3]
        self.model_frame_writer.add_image(model_frame)

        replay_frame = self.render(reduce_res=False)
        replay_frame = (255*resize(replay_frame, self.replay_shape)).astype(np.uint8)
        from ZeldaGym.modules.media import convert_to_rgb
        replay_frame = convert_to_rgb(replay_frame)
        self.replay_frame_writer.add_image(replay_frame)



    # MARK: Checks
    def check_if_done(self):
        if self.early_stopping and self.step_count > 128 and self.recent_memory.sum() < 255:
            return True
        return self.step_count > self.max_steps 
    def inventory_check(self): # find out item codes
        if self.read_m(0xDB0C) != 0 or self.read_m(0xDB0D) != 0:
            return False
        else:
            return False



    # MARK: Get
    def get_deaths(self):
        return self.bit_count(self.read_m(0xDB57))
    def bit_count(self, bits):
        return bin(bits).count("1") 
    def get_levels_sum(self):
        item_levels = [self.read_m(a) for a in item_levels_addr]
        return sum(item_levels)



    # MARK: Read Stuff
    def read_hp(self, addr):
        return self.read_m(addr)
    def read_m(self, addr):
        return self.pyboy.memory[addr]
    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit-1] == '1'


    
    # MARK: Item ID's
    def get_items_inventory(self,):
        from ZeldaGym.modules.item_ids import holding_item
        held_items = holding_item()
        return held_items