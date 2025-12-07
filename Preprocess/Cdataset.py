# Cdataset.py
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

# ---- OPTIONAL: install OpenCV if missing ----
# pip install opencv-python-headless
import cv2


# =========================
# Global config (edit here)
# =========================
IMG_SIZE   = 512

# dataset stats (for grayscale replicated to 3ch)
DS_MEAN = 0.1826
DS_STD  = 0.1647

# label normalization
MU    = 127.3207517246848
SIGMA = 41.18038858527284

# dataloader defaults
BATCH_CONTRASTIVE = 4       # can be smaller if memory tight
BATCH_REGRESSION  = 4
WORKERS = 8

# =========================
# Preprocessing & Augs
# =========================
class HandPreprocess:
    """
    Standardize radiographs:
      1) grayscale
      2) enforce common polarity (bone bright on dark)
      3) threshold + tight bbox with extra padding (to avoid cutting fingers)
      4) CLAHE
    """
    def __init__(self, pad=25, clahe_clip=2.0, tile=8, thresh=5):
        self.pad = pad
        self.clahe_clip = clahe_clip
        self.tile = tile
        self.thresh = thresh

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"))

        if arr.mean() > 128:
            arr = 255 - arr

        mask = arr > self.thresh
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(arr.shape[1] - 1, x2 + self.pad)
            y2 = min(arr.shape[0] - 1, y2 + self.pad)
            arr = arr[y1:y2 + 1, x1:x2 + 1]

        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.tile, self.tile))
        arr = clahe.apply(arr)

        arr = np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(arr)

class Masking:
    """
    Apply random block-wise masking to a tensor image.
    The mask value is 0, which corresponds to the mean after normalization.
    """
    def __init__(self, mask_ratio_range=(0.4, 0.6)):
        self.mask_ratio_range = mask_ratio_range

    def __call__(self, x):
        c, h, w = x.shape
        mask_ratio = np.random.uniform(self.mask_ratio_range[0], self.mask_ratio_range[1])
        mask_area = h * w * mask_ratio
        aspect_ratio = np.random.uniform(0.5, 2.0)
        mask_h = int(np.sqrt(mask_area * aspect_ratio))
        mask_w = int(np.sqrt(mask_area / aspect_ratio))
        mask_h = min(h, mask_h)
        mask_w = min(w, mask_w)

        if mask_h == 0 or mask_w == 0:
            return x

        top = np.random.randint(0, h - mask_h + 1)
        left = np.random.randint(0, w - mask_w + 1)

        x[:, top:top + mask_h, left:left + mask_w] = 0
        return x

def normalize_ds(mean=DS_MEAN, std=DS_STD):
    return transforms.Normalize([mean, mean, mean], [std, std, std])

def simclr_view(img_size, mean=DS_MEAN, std=DS_STD):
    return transforms.Compose([
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomApply([Masking(mask_ratio_range=(0.4, 0.6))], p=0.5),
        normalize_ds(mean, std),
    ])

def regression_train_transform(img_size, mean=DS_MEAN, std=DS_STD):
    return transforms.Compose([
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.RandomResizedCrop(img_size, scale=(0.95, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=7, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        normalize_ds(mean, std),
    ])

def regression_eval_transform(img_size, mean=DS_MEAN, std=DS_STD):
    return transforms.Compose([
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize_ds(mean, std),
    ])

class BoneAgeDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode="contrastive", img_size=IMG_SIZE, male=None):
        self.df = pd.read_csv(os.path.join("/app", csv_file), usecols=['id', 'boneage', 'male'])
        self.df['male'] = (
            self.df['male']
            .astype(str).str.strip().str.upper()
            .map({'TRUE': 1, 'FALSE': 0})
            .fillna(0)
            .astype(int)
        )

        if male is not None:
            target = int(male)
            self.df = self.df[self.df['male'] == target].reset_index(drop=True)

        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size

        if mode == "contrastive":
            self.transform1 = simclr_view(img_size, mean=DS_MEAN, std=DS_STD)
            self.transform2 = simclr_view(img_size, mean=DS_MEAN, std=DS_STD)
        elif mode == "regression_train":
            self.transform1 = regression_train_transform(img_size, mean=DS_MEAN, std=DS_STD)
            self.transform2 = self.transform1
        elif mode == "regression_eval":
            self.transform1 = regression_eval_transform(img_size, mean=DS_MEAN, std=DS_STD)
            self.transform2 = self.transform1
        else:
            raise ValueError("mode must be 'contrastive' | 'regression_train' | 'regression_eval'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['id']}.png"
        img_path = os.path.join("/app", self.root_dir, img_name)
        boneage = float(row['boneage'])
        label_norm = (boneage - MU) / SIGMA
        gender = float(row['male'])
        img = load_image(img_path)

        if self.mode == "contrastive":
            img1 = self.transform1(img)
            img2 = self.transform2(img)
        else:
            img1 = self.transform1(img)
            img2 = img1

        sample = {
            'images': img1,
            'images2': img2,
            'labels': torch.tensor([label_norm], dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.float32),
        }
        return sample

def load_image(path):
    return Image.open(path).convert('RGB')

def generate_dataset(male=None,
                     img_size=IMG_SIZE,
                     batch_contrastive=BATCH_CONTRASTIVE,
                     batch_regression=BATCH_REGRESSION,
                     workers=WORKERS):
    train_contrastive_ds = BoneAgeDataset('train.csv', 'trainimages', mode='contrastive',
                                          img_size=img_size, male=male)
    val_contrastive_ds   = BoneAgeDataset('val.csv', 'valimages',     mode='contrastive',
                                          img_size=img_size, male=male)
    train_regression_ds  = BoneAgeDataset('train.csv', 'trainimages', mode='regression_train',
                                          img_size=img_size, male=male)
    val_regression_ds    = BoneAgeDataset('val.csv', 'valimages',     mode='regression_eval',
                                          img_size=img_size, male=male)

    train_contrastive_loader = DataLoader(
        train_contrastive_ds,
        batch_size=BATCH_CONTRASTIVE, shuffle=True, num_workers=WORKERS,
        drop_last=True, pin_memory=True, persistent_workers=True
    )
    val_contrastive_loader = DataLoader(
        val_contrastive_ds,
        batch_size=batch_contrastive,
        shuffle=False,
        num_workers=workers,
        pin_memory=True, persistent_workers=True
    )
    train_regression_loader = DataLoader(
        train_regression_ds,
        batch_size=batch_regression,
        shuffle=True,
        num_workers=workers,
        pin_memory=True, persistent_workers=True,
        drop_last=True
    )
    val_regression_loader = DataLoader(
        val_regression_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True, persistent_workers=True
    )

    return (train_contrastive_ds, val_contrastive_ds,
            train_regression_ds,  val_regression_ds,
            train_contrastive_loader, val_contrastive_loader,
            train_regression_loader,  val_regression_loader)
