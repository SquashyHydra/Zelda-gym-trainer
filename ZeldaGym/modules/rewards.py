from math import floor
from numpy import zeros, uint8

# MARK: Reward Channel
def make_reward_channel(r_val, col_steps, h, w):
    #max reward value
    max_r_val = (w-1) * h * col_steps
    r_val = max(0, min(r_val, max_r_val))
    #calc row and col
    row = floor(r_val / (h * col_steps))
    memory = zeros(shape=(h, w), dtype=uint8)
    memory[:, :row] = 255
    row_covered = row * h * col_steps
    col = floor((r_val - row_covered) / col_steps)
    if row < w:
        memory[:col, row] = 255
    #Set Values
    col_covered = col * col_steps
    last_pixel = floor(r_val - row_covered - col_covered)
    if row < w and col < h:
        memory[col, row] = last_pixel * (255 // col_steps)
    return memory

# MARK: Level Reward
#Fix levels when have better understanding of levels
def calc_level_rew(reward_scale, levels_sum, max_level_rew, current_level):
    level_threash = 12
    level_sum = levels_sum
    scaled = max_level_rew
    if level_sum == level_threash and not level_sum == current_level:
        scaled = (level_sum * reward_scale) * level_threash
    elif level_sum > 0 and level_sum < level_threash and not level_sum == current_level:
        scaled = (level_sum * reward_scale) * 2.5
    elif level_sum == current_level:
        scaled = level_sum
    max_level_rew = max(max_level_rew, scaled)
    return max_level_rew 

# MARK: Tile Reward
def calc_tile_rew(map_w, seen_tiles, explore_weight, total_tile_explore):
    if map_w not in seen_tiles:
        seen_tiles.add(map_w)
        total_tile_explore += explore_weight * 0.05
    return total_tile_explore, seen_tiles

# MARK: Tile Coord Reward
def calc_tile_coords_rew(map_info, x_y, seen_tile_coords, total_coords_reward, explore_weight):
    if map_info not in seen_tile_coords:
        seen_tile_coords[map_info] = [x_y]
        total_coords_reward += round(explore_weight * 0.04, 8)
    else:
        coords = seen_tile_coords[map_info]
        if x_y not in coords:
            coords.append(x_y)
            seen_tile_coords[map_info] = coords
            total_coords_reward += round(explore_weight * 0.02, 8)
    return total_coords_reward, seen_tile_coords

# MARK: Held Item Reward
# add more rewards when more items have been collected and can be studied
def calc_held_item_rew(current_item_held, previous_item_held, reward_scale, total_held_item_rew):
    if current_item_held != previous_item_held:
        previous_item_held = current_item_held
        if current_item_held == (0, 0):
            total_held_item_rew +=  reward_scale*-1
        elif current_item_held[0] > 0 or current_item_held[1] > 0:
            total_held_item_rew += reward_scale*1
    return previous_item_held, total_held_item_rew

# MARK: Heal Reward 
def calc_heal_rew(cur_health, last_health, total_healing_rew, died_count, reward_scale):
    if cur_health > last_health and last_health > 0:
        heal_amount = cur_health - last_health
        total_healing_rew += heal_amount * reward_scale

    if cur_health == 0 and last_health > 0:
        died_count += 1
    return total_healing_rew, died_count

# MARK: Action Reward
# add the inventory button when more data about the inventory has been collected so the ai can effectively use it
def calc_action_rew(seen_tile_coords, ai_action, total_action_reward, last_seen_tile_coords, reward_scale, movement_delta=0, health_delta=0, action_penalty_scale=1.0):
    if seen_tile_coords:
        if ai_action in (4, 5):
            if health_delta < 0:
                total_action_reward += 0.004
            else:
                total_action_reward += 0.0015
        elif ai_action <= 3 and movement_delta > 0:
            total_action_reward += 0.006
        elif ai_action <= 3 and last_seen_tile_coords != seen_tile_coords:
            total_action_reward += 0.003
        elif ai_action <= 3 and movement_delta == 0:
            total_action_reward -= 0.0015 * action_penalty_scale
        elif ai_action > 5:
            total_action_reward -= 0.0005 * action_penalty_scale
        else:
            total_action_reward += 0
    return total_action_reward