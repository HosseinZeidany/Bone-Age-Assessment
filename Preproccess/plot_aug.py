import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# ✅ choose which dataset to visualize
from Cdataset import BoneAgeDataset, simclr_view, regression_train_transform, regression_eval_transform

# Example: visualize contrastive augmentations
# ds = BoneAgeDataset(csv_file='train.csv',
#                     root_dir='trainimages',
#                     mode='contrastive',
#                     img_size=512,
#                     male=None)

# or visualize regression train/eval
# ds = BoneAgeDataset('train.csv', 'trainimages', mode='regression_train', img_size=512)
ds = BoneAgeDataset('val.csv', 'valimages', mode='regression_eval', img_size=512)

# pick a few random samples
n_samples = 4
indices = np.random.choice(len(ds), n_samples, replace=False)

for i in indices:
    sample = ds[i]
    img1, img2 = sample['images'], sample['images2']
    label = sample['labels'].item() * 41.18038858527284 + 127.3207517246848  # de-normalize to months
    gender = 'Male' if sample['gender'].item() > 0.5 else 'Female'

    # convert tensors (C,H,W) → (H,W,C)
    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()

    # unnormalize for visualization
    mean, std = 0.1826, 0.1647
    img1_np = np.clip((img1_np * std) + mean, 0, 1)
    img2_np = np.clip((img2_np * std) + mean, 0, 1)

    # show side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img1_np)
    ax[0].set_title(f"img1 (Label≈{label:.1f} mo, {gender})")
    ax[1].imshow(img2_np)
    ax[1].set_title("img2 (Augmented view)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()
