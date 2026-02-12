import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from einops import rearrange

def merge_dicts_by_avg(dicts):
    sum_dict = {}
    count_dict = {}

    for dict in dicts:
        for i, j in dict.items():
            if isinstance(j, (int, float)):
                sum_dict[i] = sum_dict.get(i, 0) + j
                count_dict[i] = count_dict.get(i, 0) + 1
    
    avg_dict = {}
    for i in sum_dict:
        avg_dict[i] = sum_dict[i] / count_dict[i]

    return avg_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos if stats and len(stats) > 0]

            if all_final_infos:
                avg_infos = merge_dicts_by_avg(all_final_infos)
                for key, val in avg_infos.items():
                    self.logger.record(f"env_stats/{key}", val)

            images = self.training_env.env_method("render")
            images_arr = np.array(images)
            images_arr = images_arr[:, :, :, :3]
            images_row = rearrange(images_arr, "b h w c -> h (b w) c")
            images_row = images_row.astype(np.uint8)
            self.logger.record("trajectory/image", Image(images_row, "HWC"), exclude=("stdout", "log", "json", "csv"))

        return True