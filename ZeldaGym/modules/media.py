from os import makedirs
from pathlib import Path
from matplotlib.pyplot import imsave
from cv2 import cvtColor, COLOR_GRAY2RGB, COLOR_RGBA2RGB
from numpy import uint8

def screenshot(s_path, instance_id, reset_count, total_reward, name, render):
    ss_dir = s_path / Path('screenshots')
    makedirs(ss_dir, exist_ok=True)
    inst_dir = ss_dir / Path(f'id_{instance_id}')
    makedirs(inst_dir, exist_ok=True)
    image_save(inst_dir / Path(f'frame_{reset_count}_r{total_reward:.4f}_{instance_id}_{name}.jpeg'), render)

def image_save(dir_path, render):
    render = render[:, :, :3]
    imsave(dir_path, render)

def convert_to_rgb(image):
    if len(image.shape) == 2:  # Grayscale image
        print("Image is grayscale")
        image = cvtColor(image, COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:  # Grayscale image with single channel
            image = cvtColor(image, COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # Image is RGBA
            image = cvtColor(image, COLOR_RGBA2RGB)
    
    # Ensure image dtype is np.uint8 (if not already)
    if image.dtype != uint8:
        image = image.astype(uint8)
            
    return image