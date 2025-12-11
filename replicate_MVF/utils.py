from pathlib import Path
import cv2
from dotenv import load_dotenv
import torch


def load_env_vars():
    base = Path(__file__).resolve().parent.parent
    env_paths_file = base / ".env_paths"
    if not env_paths_file.exists():
        raise FileNotFoundError("'.env_paths' not found in this directory or any parent directories")

    # Load any variables defined directly in .env_paths (if present)
    load_dotenv(dotenv_path=env_paths_file)

def crop_center(img, crop_width, crop_height):
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    cropped = img[y1:y2, x1:x2]
    return cropped

def get_video_tensor(video_path):
    vid = cv2.VideoCapture(video_path)

    count, success = 0, True
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = torch.linspace(0, num_frames - 1, steps=16).long()
    mask = torch.zeros(num_frames, dtype=torch.bool)
    mask[indices] = True
    frames = []
    for i in range(num_frames):
        success, image = vid.read() # Read frame
        if mask[i]:
            frames.append(crop_center(image, 224, 224))

    vid.release()
    selected = torch.stack([torch.from_numpy(frame) for frame in frames])
    return selected.permute(3, 0, 1, 2).unsqueeze(0).float()/ 255.0

def tensor2cuda(data):
    """
    Put Tensor in iterable data into gpu.
    Args:
        data :(tensor or list or dict)
    """
    if type(data) == torch.Tensor:
        return data.cuda(non_blocking=True)
    elif type(data) == dict:
        keys = list(data.keys())
        for k in keys:
            data[k] = tensor2cuda(data[k])
    elif type(data) == list:
        for i in range(len(data)):
            data[i] = tensor2cuda(data[i])
    return data