import torch
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

# set seed
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
    

class CuteDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image']
        return image

    def __len__(self):
        return len(self.paths)
    

train_transform = A.Compose([
    A.Resize(height=64, width=64),
    ToTensorV2(p=1.0),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置 UNet 参数
eps_model = UNet(
    image_channels=3,
    n_channels=64,
    ch_mults=[1, 2, 2, 4],
    is_attn=[False, False, False, True],
).to(device)

# 配置 DDPM参数
n_steps = 1000
diffusion = DenoiseDiffusion(
    eps_model=eps_model,
    n_steps=n_steps,
    device=device,
)
# 图片路径
image_dir = ".../"
# 每个epoch看生成的效果图目录
look_look_dir = ".../"
# 模型保存目录
save_model_dir = ".../"

# 训练参数
image_paths = [image_dir + p for p in os.listdir(image_dir)]
image_size = 64
batch_size = 64
learning_rate = 2e-5
epoch = 20
n_samples = 4
dataset = CuteDataset(paths=image_paths, transform=train_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
optimizer = torch.optim.Adam(eps_model.parameters(), lr=learning_rate)

for e in range(epoch):
    for i, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = diffusion.loss(data)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {e+1}/{epoch}, Batch: {i+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

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

        for i, x_i in enumerate(x):
            x_i = x_i.transpose(1, 2, 0)
            x_i *= 255
            x_i[x_i<0]=0
            x_i[x_i>255]=255
            x_i = x_i.astype(np.uint8)
            cv2.imwrite(f"{look_look_dir}{i}.png", x_i)
    # if epoch % 1 == 0:
    torch.save(eps_model.state_dict(), f"{save_model_dir}eps_model_{e+1}.pth")
