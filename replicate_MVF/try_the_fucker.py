from tadaconv.utils.config import Config
from tadaconv.models.base.builder import build_model
import torch
import os
import tadaconv.utils.checkpoint as cu
from utils import load_env_vars, get_video_tensor
import logging
logging.basicConfig(level=logging.INFO)

load_env_vars()

labels_path = os.environ.get("K710Labels", "") 
with open(labels_path, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(labels)} labels")
labels.sort()

video_tensor1 = get_video_tensor(os.environ.get("EXAMPLE_VIDEO_1", ""))
video_tensor2: torch.Tensor = get_video_tensor(os.environ.get("EXAMPLE_VIDEO_2", ""))

video_tensor = torch.cat([video_tensor1, video_tensor2], dim=0).unsqueeze(0)

video_tensor = torch.cat([video_tensor, video_tensor], dim=0)  # (B, V, C, T, H, W)

print(f"Input video tensor shape: {video_tensor.shape}")
exit()

assert (video_tensor.shape[0], video_tensor.shape[1]) == (2, 2), "Expected 2 views in the input tensor"

cfg = Config(load=True)
model, _ = build_model(cfg)
cu.load_test_checkpoint(cfg, model, None, None)


print("Model loaded.")


mask = torch.ones((video_tensor.shape[0], video_tensor.shape[1]), dtype=torch.long)
mask[0,1] = 0

model.eval()
with torch.no_grad():
    out = model(video_tensor, mask=mask)

assert out[0]['severity'].shape[0] == mask.shape[0], "Output batch size does not match input batch size"
print(f"Output shape: {out[0]['severity'].shape}")
k = min(10, out[0]['severity'].shape[1])
values, indices = torch.topk(out[0]['severity'], k=k, dim=1)
for rank, (v, i) in enumerate(zip(values[0], indices[0]), start=1):
    print(f"{rank}: {v.item():.6f} - {labels[i.item()]}")

