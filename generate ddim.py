import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
import cv2
import random
import albumentations as A

from albumentations.pytorch import ToTensorV2
from unet import UNet
from ddpm import DenoiseDiffusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps_model = UNet(
    image_channels=3,
    n_channels=64,
    ch_mults=[1, 2, 1, 2],
    is_attn=[False, False, False, True],
)
model_path = "/home/jinfan/Desktop/gsxm/SD/ddpm/checkpoints/eps_model_19.pth"
out_dir = "/home/jinfan/Desktop/gsxm/SD/ddpm/data/look2/"
eps_model.load_state_dict(torch.load(model_path))
eps_model.to(device)
eps_model.eval()
n_steps = 1000
prev_steps = 5

diffusion = DenoiseDiffusion(
    eps_model=eps_model,
    n_steps=n_steps,
    device=device,
)
image_size = 64
n_samples = 4
with torch.no_grad():
    x = torch.randn([n_samples, 3, image_size, image_size],
                    device=device)
    # Remove noise for $T$ steps
    for t_ in range(n_steps//prev_steps-1):
        # $t$
        t = (n_steps//prev_steps - t_ -1)*prev_steps
        print(t)
        x = diffusion.ddim_sample(x, torch.full((n_samples,), t, dtype=torch.long, device=device), torch.full((n_samples,), t-prev_steps, dtype=torch.long, device=device))
    x = x.cpu().numpy()
    for i, x_i in enumerate(x):
        x_i = x_i.transpose(1, 2, 0)
        x_i *= 255
        x_i[x_i<0]=0
        x_i[x_i>255]=255
        x_i = x_i.astype(np.uint8)
        x_i = cv2.cvtColor(x_i, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{out_dir}{i}.png", x_i)
