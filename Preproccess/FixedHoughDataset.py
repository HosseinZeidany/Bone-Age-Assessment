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
BATCH_CONTRASTIVE = 16       # can be smaller if memory tight
BATCH_REGRESSION  = 16
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


class HoughCircleMasking:
    """
    Stable Hough masking:
      - max 2 circles
      - masking prob = 0.2
      - masking before Normalize
      - safe: uses x.clone()
    """
    def __init__(self, num_circles_to_mask=2, p=0.2,
                 min_r=12, max_r=40):
        self.num_circles = num_circles_to_mask
        self.p = p
        self.min_r = min_r
        self.max_r = max_r

    def __call__(self, x):
        # ---------------------------------------------
        # x is a tensor [C,H,W] in range [0,1] (after ToTensor)
        # ---------------------------------------------
        if np.random.rand() > self.p:
            return x   # no masking this time

        x = x.clone()    # avoid in-place touching pipeline tensor

        # convert to uint8 numpy
        np_img = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=self.min_r,
            maxRadius=self.max_r
        )

        if circles is None:
            return x

        circles = np.uint16(np.around(circles[0]))
        n = circles.shape[0]

        # pick up to 2 circles safely
        k = min(self.num_circles, n)
        idx = np.random.choice(n, k, replace=False)

        C, H, W = x.shape

        for i in idx:
            cx, cy, r = circles[i]

            # build mask in numpy
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)

            mask_tensor = torch.from_numpy(mask).bool().to(x.device)
            mask_tensor = mask_tensor.unsqueeze(0).expand_as(x)

            x[mask_tensor] = 0.0  # zero-out circle

        return x

def normalize_ds(mean=DS_MEAN, std=DS_STD):
    return transforms.Normalize([mean, mean, mean], [std, std, std])

from torchvision import transforms

def contrastive_view(img_size, mean=DS_MEAN, std=DS_STD,
                     use_masking=False, mask_p=0.25, mask_num=3):
    """
    Returns a transform. If use_masking=True, HoughCircleMasking is applied
    AFTER ToTensor() and BEFORE Normalize(), with internal probability mask_p.
    """
    aug_list = [
        HandPreprocess(pad=25, clahe_clip=2.0, tile=8, thresh=5),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ]

    # insert masking only if requested
    if use_masking:
        # Wrap HoughCircleMasking in RandomApply to keep control here.
        hough = HoughCircleMasking(num_circles_to_mask=mask_num, p=mask_p)
        # transforms.RandomApply accepts callables that take tensors, so this is fine.
        aug_list.append(transforms.RandomApply([hough], p=1.0))

    aug_list.append(transforms.Normalize([mean, mean, mean], [std, std, std]))
    return transforms.Compose(aug_list)


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
        self.df = pd.read_csv(csv_file, usecols=['id', 'boneage', 'male'])
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
            self.transform1 = contrastive_view(img_size, mean=DS_MEAN, std=DS_STD, use_masking=True,mask_p=0.3,mask_num=3)
            self.transform2 = contrastive_view(img_size, mean=DS_MEAN, std=DS_STD, use_masking=True,mask_p=0.1,mask_num=2)
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
            'labels_raw': torch.tensor(boneage, dtype=torch.float32),
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
                                          img_size=img_size)
    val_contrastive_ds   = BoneAgeDataset('val.csv', 'valimages',     mode='contrastive',
                                          img_size=img_size)
    train_regression_ds  = BoneAgeDataset('train.csv', 'trainimages', mode='regression_train',
                                          img_size=img_size)
    val_regression_ds    = BoneAgeDataset('val.csv', 'valimages',     mode='regression_eval',
                                          img_size=img_size)
    # test dataset (NO augmentation)
    test_regression_ds = BoneAgeDataset(
        csv_file="test.csv",
        root_dir="testimages",
        mode="regression_eval",
        img_size=IMG_SIZE,
        male=None
    )

    test_regression_loader = DataLoader(
        test_regression_ds,
        batch_size=1,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

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
            train_regression_ds,  val_regression_ds,test_regression_ds,
            train_contrastive_loader, val_contrastive_loader,
            train_regression_loader,  val_regression_loader,test_regression_loader)
