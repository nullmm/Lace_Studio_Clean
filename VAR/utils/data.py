import os.path as osp
import os
import glob
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import transforms

def normalize_01_into_pm1(x):  # 将 x 从 [0, 1] 归一化到 [-1, 1]
    return x.add(x).add_(-1)

# ----------------------------------------------------
# 🌟 核心修改：无条件蕾丝生成数据集 (Unconditional Lace Dataset)
# ----------------------------------------------------
class UncondLaceDataset(Dataset):
    def __init__(self, data_path, transform=None, hflip=False, final_reso=512):
        self.data_path = data_path
        self.transform = transform
        self.hflip = hflip
        self.final_reso = final_reso
        # 现在我们只需要寻找原图 _rgb.png，彻底抛弃红绿掩膜图！
        self.rgb_files = sorted(glob.glob(os.path.join(data_path, "*_rgb.png")))
        if len(self.rgb_files) == 0:
            raise ValueError(f"在 {data_path} 目录下没有找到 *_rgb.png 文件！")

    def __len__(self):
        return len(self.rgb_files)

    # 在 VAR/utils/data.py 的 __getitem__ 方法中修改：


    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        cond_path = rgb_path.replace("_rgb.png", "_cond.png") # 找回它的红绿兄弟

        img_rgb = Image.open(rgb_path).convert('RGB')
        img_cond = Image.open(cond_path).convert('RGB') # 🌟 真正读取红绿图

        img_rgb = TF.resize(img_rgb, (self.final_reso, self.final_reso), interpolation=TF.InterpolationMode.BILINEAR)
        img_cond = TF.resize(img_cond, (self.final_reso, self.final_reso), interpolation=TF.InterpolationMode.NEAREST) # 掩膜最好用最近邻插值

        if self.hflip and random.random() > 0.5:
            img_rgb = TF.hflip(img_rgb)
            img_cond = TF.hflip(img_cond)

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_cond = self.transform(img_cond) # 🌟 把真实的掩膜也转成 Tensor 喂给模型

        return img_rgb, img_cond


# ----------------------------------------------------
# 接入 UncondLaceDataset
# ----------------------------------------------------
def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    base_transform = transforms.Compose([
        transforms.ToTensor(), 
        normalize_01_into_pm1,
    ])
    
    # 实例化无条件数据集
    dataset = UncondLaceDataset(data_path=data_path, transform=base_transform, hflip=hflip, final_reso=final_reso)
    
    train_set = dataset
    val_set = dataset 
    
    num_classes = 1 
    
    print(f'[Dataset] Unconditional Lace Dataset Loaded: {len(train_set)} images.')
    print_aug(base_transform, '[train/val]')
    
    return num_classes, train_set, val_set


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')