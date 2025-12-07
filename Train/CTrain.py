import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
#from torchvision import models
from daaata import generate_dataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
MU    = 127.3207517246848
SIGMA = 41.18038858527284

class EarlyStopping:
    """
    Stop training when the monitored metric stops improving.
    mode='min' for losses, mode='max' for accuracies/R^2.
    """
    def __init__(self, patience=10, mode='min', delta=0.0, checkpoint_path='best.pt', verbose=True):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.best = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.should_stop = False

    def __call__(self, value, model):
        improved = (value < self.best - self.delta) if self.mode == 'min' else (value > self.best + self.delta)
        if improved:
            self.best = value
            self.counter = 0
            # save checkpoint
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"[EarlyStopping] New best ({self.mode}): {self.best:.6f}. Saved to {self.checkpoint_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience}).")
            if self.counter >= self.patience:
                self.should_stop = True


class ContrastiveModel(nn.Module):
    def __init__(self, proj_dim=128, hidden_dim=1024):
        super(ContrastiveModel, self).__init__()

        # --- ConvNeXt-Tiny backbone ---
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        self.backbone = backbone.features          # feature extractor
        self.out_dim = 768                         # ConvNeXt-Tiny output channels

        # --- Projection head (SimCLR style, smaller for stability) ---
        self.projector = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False)
        )

    def forward_features(self, x):
        """Return encoder features before projection (for downstream tasks)."""
        feats = self.backbone(x)
        feats = feats.mean([-2, -1])
        return feats

    def forward_projection(self, x):
        """Return contrastive projection (for SimCLR loss)."""
        feats = self.forward_features(x)
        z = self.projector(feats)
        z = F.normalize(z, p=2, dim=1)
        return z

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss used in SimCLR-style contrastive learning.

    Args:
        z1, z2 : torch.Tensor
            Normalized embedding vectors of shape (batch_size, projection_dim)
            for the two augmented views of the same images.
        temperature : float
            Scaling factor that controls how sharply the similarities are distributed.

    Returns:
        torch.Tensor : scalar loss value
    """

    B = z1.size(0)  # Number of samples in the batch

    # Concatenate embeddings for both views -> shape (2B, projection_dim)
    z = torch.cat([z1, z2], dim=0)

    # Compute the cosine similarity matrix between all pairs (2B x 2B)
    # Since z is L2-normalized, dot product == cosine similarity
    sim = torch.matmul(z, z.t())

    # Create a mask to ignore self-similarities along the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)  # Replace diagonal with a large negative number

    # Build labels: positive pair for i-th sample is (i + B) and vice versa
    # For the concatenated batch [z1; z2], positives are across the two halves
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)

    # Scale similarities by temperature to control sharpness of distribution
    logits = sim / temperature

    # Numerical stability trick: subtract max(logits) per row before softmax
    logits = logits - logits.max(dim=1, keepdim=True).values

    # Compute cross-entropy loss; each sample’s positive pair is its label
    loss = F.cross_entropy(logits, pos)

    return loss

@torch.no_grad()
def eval_contrastive(model, val_loader, device, temperature=0.5):
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        x1 = batch['images'].to(device)
        x2 = batch['images2'].to(device)
        z1 = model.forward_projection(x1)
        z2 = model.forward_projection(x2)
        loss = nt_xent_loss(z1, z2, temperature)
        total += loss.item()
        n += 1
    return total / max(n,1)

def train_contrastive(model, optimizer, train_loader, val_loader, device,
                      epochs=50, temperature=0.2, patience=15, ckpt='best_contrastive.pt',
                      writer: SummaryWriter=None, scheduler=None):

    stopper = EarlyStopping(patience=patience, mode='min', checkpoint_path=ckpt)
    global_step = 0
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_hist = []

        # ---- tqdm progress bar (training) ----
        progress = tqdm(total=len(train_loader), desc=f"Contrastive Epoch {epoch}", ncols=110)
        for iter_num, batch in enumerate(train_loader):
            x1 = batch['images'].to(device)
            x2 = batch['images2'].to(device)

            z1 = model.forward_projection(x1)
            z2 = model.forward_projection(x2)
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_hist.append(loss.item())

            progress.set_description(
                desc=f"Contrastive - Ep: {epoch:03d} | It: {iter_num:04d} | Loss: {np.mean(train_loss_hist):.4f}"
            )
            progress.update(1)

            if writer is not None:
                writer.add_scalar("train/contrastive_iter_loss", loss.item(), global_step)
            global_step += 1

        progress.close()
        train_loss = np.mean(train_loss_hist)
        tqdm.write(f"Contrastive - Ep: {epoch} | Loss: {train_loss:.4f}")

        # ---- validation (no bar spam; keep simple) ----
        val_loss = eval_contrastive(model, val_loader, device, temperature)
        tqdm.write(f"Contrastive Val - Ep: {epoch} | Loss: {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("epoch/train_contrastive_loss", train_loss, epoch)
            writer.add_scalar("epoch/val_contrastive_loss", val_loss, epoch)
            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group_{i}", pg.get('lr', 0.0), epoch)

        stopper(val_loss, model)
        if stopper.should_stop:
            tqdm.write("[EarlyStopping] Stopping contrastive training.")
            break

        if val_loss < best_val:
            best_val = val_loss

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model

def load_contrastive_from_ckpt(ckpt_path, device):
    """
    Load the contrastive model (encoder+projector) exactly as it was pretrained.
    Make sure proj_dim/hidden_dim match your contrastive training config.
    """
    model = ContrastiveModel(proj_dim=128, hidden_dim=1024).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"[OK] Loaded contrastive checkpoint: {ckpt_path}")
    return model

@torch.no_grad()
def eval_regression(model, val_loader, device):
    model.eval()
    total_abs_norm, n_items = 0.0, 0
    for batch in val_loader:
        x = batch['images'].to(device)
        y = batch['labels'].to(device)   # normalized
        g = batch['gender'].to(device)
        pred = model(x, g)               # normalized
        abs_err = (pred - y).abs()       # [B,1]
        total_abs_norm += abs_err.sum().item()
        n_items += abs_err.numel()
    mean_mae_norm = total_abs_norm / max(n_items, 1)
    return mean_mae_norm * SIGMA  # months


class BoneAgePredictionModel(nn.Module):
    def __init__(self, contrastive_model, num_classes=1):
        super(BoneAgePredictionModel, self).__init__()
        self.contrastive_model = contrastive_model

        # NEW: stabilize ConvNeXt feature scale before the head
        self.pre_head_norm = nn.LayerNorm(768, eps=1e-6)

        self.regression_head = nn.Sequential(
            nn.Linear(768 + 1, 256),  # 768 encoder dims + 1 gender
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, gender):
        # 1) get encoder features from ConvNeXt
        emb = self.contrastive_model.forward_features(x)   # may be [B,768] or [B,768,H,W]

        # 2) GAP if it's a feature map (ConvNeXt forward_features usually returns [B, C, H, W])
        if emb.dim() == 4:
            emb = emb.mean(dim=(-2, -1))                   # [B,768]

        # 3) LayerNorm to match ConvNeXt classifier pre-norm
        emb = self.pre_head_norm(emb)                      # [B,768]

        # 4) concat gender and predict
        g = gender.view(-1, 1).float()                     # [B,1]
        xcat = torch.cat([emb, g], dim=1)                  # [B,769]
        return self.regression_head(xcat)                  # [B,1]


def train_bone_age_model(model, optimizer, train_loader, val_loader, device,
                         epochs=50, patience=20, ckpt='best_regression.pt',
                         writer=None, scheduler=None):

    stopper = EarlyStopping(patience=patience, mode='min', checkpoint_path=ckpt)
    global_step = 0
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        # store batch MAE in months for display
        train_mae_months_hist = []

        progress = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch}", ncols=110)
        for iter_num, batch in enumerate(train_loader):
            x = batch['images'].to(device)
            y = batch['labels'].to(device)   # normalized labels
            g = batch['gender'].to(device)

            pred = model(x, g).view(-1, 1)   # normalized prediction
            y = y.view(-1, 1)

            # optimize with normalized MAE (stable)
            loss_norm = F.l1_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss_norm.backward()
            optimizer.step()

            # for reporting: convert batch MAE to months
            mae_months_batch = loss_norm.item() * SIGMA
            train_mae_months_hist.append(mae_months_batch)

            progress.set_description(
                desc=f"Train - Ep: {epoch:03d} | It: {iter_num:04d} | MAE (mo): {np.mean(train_mae_months_hist):.2f}"
            )
            progress.update(1)

            if writer is not None:
                # log both if you want; here we log months
                writer.add_scalar("train/regression_iter_mae_months", mae_months_batch, global_step)
            global_step += 1

        progress.close()
        train_mae_months = float(np.mean(train_mae_months_hist))
        tqdm.write(f"Train - Ep: {epoch} | MAE (months): {train_mae_months:.2f}")

        # ---------- Validation in months ----------
        val_mae_months = eval_regression(model, val_loader, device)
        tqdm.write(f"Val   - Ep: {epoch} | MAE (months): {val_mae_months:.2f}")

        if writer is not None:
            writer.add_scalar("epoch/train_regression_mae_months", train_mae_months, epoch)
            writer.add_scalar("epoch/val_regression_mae_months", val_mae_months, epoch)
            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group_{i}", pg.get('lr', 0.0), epoch)

        # Early stopping monitors MAE in months
        stopper(val_mae_months, model)
        if stopper.should_stop:
            tqdm.write("[EarlyStopping] Stopping regression training.")
            break

        if val_mae_months < best_val:
            best_val = val_mae_months

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


# Initialize the model and optimizer
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build loaders (you already have generate_dataset)
    (
        train_contrastive_ds,
        val_contrastive_ds,
        train_regression_ds,
        val_regression_ds,
        train_contrastive_loader,
        val_contrastive_loader,
        train_regression_loader,
        val_regression_loader
    ) = generate_dataset(male=None)  # or male=True / False to filter

    # Writers
    tb_contrastive = SummaryWriter(log_dir="runs/contrastive")
    tb_regression = SummaryWriter(log_dir="runs/regression")

    # contrastive pretrain
    contrastive_model = ContrastiveModel(proj_dim=128, hidden_dim=1024).to(device)
    opt_c = torch.optim.AdamW(contrastive_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=50)
    contrastive_model = train_contrastive(
        contrastive_model, opt_c, train_contrastive_loader, val_contrastive_loader, device,
        epochs=50, temperature=0.5, patience=15, ckpt='best_contrastive.pt',
        writer=tb_contrastive, scheduler=scheduler_c
    )
    # # fine-tune regression head
    # reg_model = BoneAgePredictionModel(contrastive_model).to(device)

    # ---- load pretrained contrastive encoder ----
    # CONTRASTIVE_CKPT = "best_contrastive.pt"  # adjust path if needed
    # contrastive_model = load_contrastive_from_ckpt(CONTRASTIVE_CKPT, device)

    # ---- build regression model on top of the encoder features (h ⊕ gender) ----
    reg_model = BoneAgePredictionModel(contrastive_model).to(device)
    print(reg_model)

    # summary(reg_model, input_size=[(3, 512, 512),(1,)],batch_size=4)
    # A) head expects 768+1
    print("[Sanity] reg in_features:", reg_model.regression_head[0].in_features)  # expect 769

    # B) encoder is trainable
    print("[Sanity] trainable encoder params:",
          sum(p.requires_grad for p in reg_model.contrastive_model.parameters()))  # must be > 0

    # C) optimizer built on the final model

    # ---- end-to-end fine-tuning (NO freeze/unfreeze, NO warm-up) ----
    opt_r = torch.optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)
    print("[Sanity] #params in optimizer:",
          sum(p.numel() for g in opt_r.param_groups for p in g['params']))
    scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=50)
    reg_model = train_bone_age_model(
        reg_model, opt_r, train_regression_loader, val_regression_loader, device,
        epochs=50, patience=20, ckpt='best_regression.pt',
        writer=tb_regression, scheduler=scheduler_r
    )

    tb_contrastive.close()
    tb_regression.close()