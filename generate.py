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
    for t_ in range(n_steps):
        # $t$
        t = n_steps - t_ - 1
        # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
    x = x.cpu().numpy()
    # np.save(f"/home/jinfan/Desktop/ai_tt1/SD/DDPM/data/look/samples_{e+1}.npy", x)
    for i, x_i in enumerate(x):
        x_i = x_i.transpose(1, 2, 0)
        x_i *= 255
        x_i[x_i<0]=0
        x_i[x_i>255]=255
        x_i = x_i.astype(np.uint8)
        cv2.imwrite(f"{out_dir}{i}.png", x_i)
