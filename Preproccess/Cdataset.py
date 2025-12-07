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
        self.pad = pad                 # ↑ was 12 → 25 (more margin)
        self.clahe_clip = clahe_clip
        self.tile = tile
        self.thresh = thresh           # ↓ was 10 → 5 (keep faint fingertips)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"))

        # polarity: invert if background is bright
        if arr.mean() > 128:
            arr = 255 - arr

        # threshold + bbox with padding
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

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.tile, self.tile))
        arr = clahe.apply(arr)

        # back to 3ch PIL
        arr = np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(arr)


def normalize_ds(mean=DS_MEAN, std=DS_STD):
    return transforms.Normalize([mean, mean, mean], [std, std, std])


# --------- Contrastive views (SimCLR-style, strong + stochastic) ----------
def simclr_view(img_size, mean=DS_MEAN, std=DS_STD):
    # k = int(0.1 * img_size)
    # if k % 2 == 0: k += 1
    return transforms.Compose([
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        normalize_ds(mean, std),
    ])

# --------- Regression train (light, anatomy-safe) ----------
def regression_train_transform(img_size, mean=DS_MEAN, std=DS_STD):
    return transforms.Compose([
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.RandomResizedCrop(img_size, scale=(0.95, 1.0)),  # gentler crop
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=7, translate=(0.02, 0.02), scale=(0.98, 1.02)),  # tiny affine
        transforms.ToTensor(),
        normalize_ds(mean, std),
    ])


# --------- Regression eval (deterministic) ----------
def regression_eval_transform(img_size, mean=DS_MEAN, std=DS_STD):
    return transforms.Compose([
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize_ds(mean, std),
    ])


# =========================
# Dataset
# =========================
class BoneAgeDataset(Dataset):
    """
    mode:
      - 'contrastive'      -> returns two views: images, images2 (same image, different augs)
      - 'regression_train' -> returns images (light augs), images2 (same as images, unused)
      - 'regression_eval'  -> returns deterministic images (no random augs)
    """
    def __init__(self, csv_file, root_dir, mode="contrastive", img_size=IMG_SIZE, male=None):
        self.df = pd.read_csv(csv_file, usecols=['id', 'boneage', 'male'])
        # Normalize 'male' to 0/1 (handles TRUE/FALSE with spaces, etc.)
        self.df['male'] = (
            self.df['male']
            .astype(str).str.strip().str.upper()
            .map({'TRUE': 1, 'FALSE': 0})
            .fillna(0)
            .astype(int)
        )

        # Keep the gender filter (optional: pass male=True / False / 1 / 0)
        if male is not None:
            # accept True/False or 1/0
            target = int(male)
            self.df = self.df[self.df['male'] == target].reset_index(drop=True)

        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size

        # pick transforms based on mode
        if mode == "contrastive":
            # keep your original transforms
            self.transform1 = simclr_view(img_size, mean=DS_MEAN, std=DS_STD)
            self.transform2 = simclr_view(img_size, mean=DS_MEAN, std=DS_STD)

            # NOTE: self.pre1 and self.pre2 should both be HandPreprocess; we will run just once
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
        img_path = os.path.join(self.root_dir, img_name)

        # label normalize
        boneage = float(row['boneage'])
        label_norm = (boneage - MU) / SIGMA

        # gender as 0/1 float
        gender = float(row['male'])

        # load + transform

        img = load_image(img_path)

        if self.mode == "contrastive":
            # run HandPreprocess ONCE (use pre1; pre2 is the same class)
            img1 = self.transform1(img)
            img2 = self.transform2(img)
        else:
            img1 = self.transform1(img)
            img2 = img1  # regression: second view unused

        sample = {
            'images': img1,
            'images2': img2,
            'labels': torch.tensor([label_norm], dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.float32),
        }
        return sample

def load_image(path):
    return Image.open(path).convert('RGB')


# =========================
# Loader factory
# =========================
def generate_dataset(male=None,
                     img_size=IMG_SIZE,
                     batch_contrastive=BATCH_CONTRASTIVE,
                     batch_regression=BATCH_REGRESSION,
                     workers=WORKERS):
    """
    Returns:
      train_contrastive_ds, val_contrastive_ds, train_regression_ds, val_regression_ds,
      train_contrastive_loader, val_contrastive_loader,
      train_regression_loader,  val_regression_loader
    """
    # datasets
    train_contrastive_ds = BoneAgeDataset('train.csv', 'trainimages', mode='contrastive',
                                          img_size=img_size, male=male)
    val_contrastive_ds   = BoneAgeDataset('val.csv', 'valimages',     mode='contrastive',
                                          img_size=img_size, male=male)

    train_regression_ds  = BoneAgeDataset('train.csv', 'trainimages', mode='regression_train',
                                          img_size=img_size, male=male)
    val_regression_ds    = BoneAgeDataset('val.csv', 'valimages',     mode='regression_eval',
                                          img_size=img_size, male=male)

    # loaders
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
