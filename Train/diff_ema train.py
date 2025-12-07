# CTrain.py (drop-in)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from Preprocess.Cdataset import generate_dataset, SIGMA

# ================= EarlyStopping =================
class EarlyStopping:
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
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"[EarlyStopping] New best ({self.mode}): {self.best:.6f}. Saved to {self.checkpoint_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience}).")
            if self.counter >= self.patience:
                self.should_stop = True


# ================= Encoder (ResNet-34) =================
class Encoder(nn.Module):
    def __init__(self, proj_dim=128, hidden_dim=1024):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

        self.projector = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward_features(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x):
        h = self.forward_features(x)
        z = self.projector(h)
        return F.normalize(z, p=2, dim=1)


# ================= MoCo =================
class MoCo(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = Encoder(proj_dim=dim)
        self.encoder_k = Encoder(proj_dim=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        self._dequeue_and_enqueue(k)
        return F.cross_entropy(logits, labels)


@torch.no_grad()
def eval_contrastive(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        x1 = batch['images'].to(device, non_blocking=True)
        x2 = batch['images2'].to(device, non_blocking=True)
        loss = model(x1, x2)
        total += loss.item()
        n += 1
    return total / max(n, 1)

def train_contrastive(model, optimizer, train_loader, val_loader, device,
                      epochs=50, patience=15, ckpt='best_contrastive.pt',
                      writer: SummaryWriter=None, scheduler=None):

    stopper = EarlyStopping(patience=patience, mode='min', checkpoint_path=ckpt)
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_hist = []
        progress = tqdm(total=len(train_loader), desc=f"Contrastive Epoch {epoch}", ncols=110)

        for it, batch in enumerate(train_loader):
            x1 = batch['images'].to(device, non_blocking=True)
            x2 = batch['images2'].to(device, non_blocking=True)

            loss = model(x1, x2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(loss.detach().item())
            train_loss_hist.append(loss_val)

            progress.set_description(
                desc=f"Contrastive - Ep:{epoch:03d} | It:{it:04d} | Loss:{np.mean(train_loss_hist):.4f}"
            )
            progress.update(1)

            if writer is not None:
                writer.add_scalar("train/batch_contrastive_loss", loss_val, global_step)
            global_step += 1

        progress.close()
        train_loss = float(np.mean(train_loss_hist))
        tqdm.write(f"Contrastive - Ep:{epoch} | Train Loss: {train_loss:.4f}")

        val_loss = eval_contrastive(model, val_loader, device)
        tqdm.write(f"Contrastive - Ep:{epoch} | Val Loss: {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("epoch/train_contrastive_loss", train_loss, epoch)
            writer.add_scalar("epoch/val_contrastive_loss", val_loss, epoch)
            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group_{i}", pg.get('lr', 0.0), epoch)

        stopper(val_loss, model)
        if stopper.should_stop:
            tqdm.write("[EarlyStopping] Stopping contrastive training.")
            break

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model



# ================= Regression model (plain deeper head) =================
class BoneAgePredictionModel(nn.Module):
    def __init__(self, contrastive_model, num_classes=1):
        super().__init__()
        self.contrastive_model = contrastive_model
        self.pre_head_norm = nn.LayerNorm(512, eps=1e-6)

        in_dim = 512 + 1
        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.03),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, num_classes)
        )

    def forward(self, x, gender):
        h = self.contrastive_model.forward_features(x)
        h = self.pre_head_norm(h)
        g = gender.view(-1, 1).float()
        xcat = torch.cat([h, g], dim=1)
        return self.regression_head(xcat)


# ================= Eval: true item-wise MAE (months) =================
@torch.no_grad()
def eval_regression(model, val_loader, device, tta: bool = False):
    model.eval()
    total_abs_norm, n_items = 0.0, 0

    for batch in val_loader:
        x = batch['images'].to(device, non_blocking=True)
        y = batch['labels'].to(device, non_blocking=True)
        g = batch['gender'].to(device, non_blocking=True)

        if not tta:
            pred = model(x, g)
        else:
            pred1 = model(x, g)
            x_flip = torch.flip(x, dims=[3])
            pred2 = model(x_flip, g)
            pred = 0.5 * (pred1 + pred2)

        abs_err = (pred - y).abs()
        total_abs_norm += abs_err.sum().item()
        n_items += abs_err.numel()

    mean_mae_norm = total_abs_norm / max(n_items, 1)
    mean_mae_months = mean_mae_norm * SIGMA
    return mean_mae_months


# ================= Train regression (Huber + EMA + Cosine) =================
def train_bone_age_model(model, optimizer, train_loader, val_loader, device,
                         epochs=50, patience=20, ckpt='best_regression.pt',
                         writer=None, scheduler=None):

    BETA_MONTHS = 1.5
    BETA_NORM = BETA_MONTHS / SIGMA

    stopper = EarlyStopping(patience=patience, mode='min', checkpoint_path=ckpt)
    global_step = 0

    from copy import deepcopy
    model_ema = deepcopy(model).to(device)
    for p in model_ema.parameters():
        p.requires_grad_(False)

    EMA_START_EPOCH = 5

    def ema_update(m_src, m_tgt, decay):
        with torch.no_grad():
            sd, sd_ema = m_src.state_dict(), m_tgt.state_dict()
            for k in sd_ema.keys():
                v = sd[k]
                if v.dtype.is_floating_point:
                    sd_ema[k].mul_(decay).add_(v, alpha=1.0 - decay)
                else:
                    sd_ema[k] = v

    for epoch in range(1, epochs + 1):
        model.train()
        train_mae_months_hist = []
        progress = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch}", ncols=110)

        if epoch < 20:
            ema_decay = 0.99
        elif epoch < 40:
            ema_decay = 0.995
        else:
            ema_decay = 0.999

        for it, batch in enumerate(train_loader):
            x = batch['images'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            g = batch['gender'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, g).view(-1, 1)
            y = y.view(-1, 1)

            loss_norm = F.smooth_l1_loss(pred, y, beta=BETA_NORM)
            loss_norm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch >= EMA_START_EPOCH:
                ema_update(model, model_ema, decay=ema_decay)

            mae_months_batch = loss_norm.item() * SIGMA
            train_mae_months_hist.append(mae_months_batch)

            progress.set_description(
                desc=f"Train - Ep:{epoch:03d} | It:{it:04d} | MAE(mo): {np.mean(train_mae_months_hist):.2f}"
            )
            progress.update(1)

            if writer is not None:
                writer.add_scalar("train/batch_mae_months", mae_months_batch, global_step)
                writer.add_scalar("train/batch_loss_norm", loss_norm.item(), global_step)
            global_step += 1

        progress.close()

        train_mae_months = float(np.mean(train_mae_months_hist))
        tqdm.write(f"Train - Ep:{epoch} | MAE (months): {train_mae_months:.2f}")

        val_raw_no_tta = eval_regression(model, val_loader, device, tta=False)
        val_raw_tta = eval_regression(model, val_loader, device, tta=True)

        val_ema_no_tta = eval_regression(model_ema, val_loader, device, tta=False)
        val_ema_tta = eval_regression(model_ema, val_loader, device, tta=True)

        tqdm.write(
            f"Val - Ep:{epoch} | "
            f"RAW(noTTA): {val_raw_no_tta:.2f}  | RAW(TTA): {val_raw_tta:.2f}  || "
            f"EMA(noTTA): {val_ema_no_tta:.2f}  | EMA(TTA): {val_ema_tta:.2f}"
        )

        if writer is not None:
            writer.add_scalar("val/raw_no_tta", val_raw_no_tta, epoch)
            writer.add_scalar("val/raw_tta", val_raw_tta, epoch)
            writer.add_scalar("val/ema_no_tta", val_ema_no_tta, epoch)
            writer.add_scalar("val/ema_tta", val_ema_tta, epoch)

        stopper(val_ema_tta, model_ema)

        if stopper.should_stop:
            tqdm.write("[EarlyStopping] Stopping regression training.")
            break

        if scheduler is not None:
            scheduler.step()

    torch.save(model_ema.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


# ================= Contrastive loader (optional) =================
def load_contrastive_from_ckpt(ckpt_path, device):
    model = MoCo().to(device)
    if ckpt_path is None:
        print("[Info] No contrastive checkpoint provided. Using ImageNet-initialized encoder.")
        return model
    try:
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded contrastive checkpoint (strict=False): {ckpt_path}")
        if missing or unexpected:
            print("[load] missing keys:", missing)
            print("[load] unexpected keys:", unexpected)
    except Exception as e:
        print(f"[WARN] Could not load {ckpt_path}: {e}\nUsing ImageNet weights instead.")
    return model


# ================= Main =================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[CUDA] available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[CUDA] name:", torch.cuda.get_device_name(0))

    (
        train_contrastive_ds,
        val_contrastive_ds,
        train_regression_ds,
        val_regression_ds,
        train_contrastive_loader,
        val_contrastive_loader,
        train_regression_loader,
        val_regression_loader
    ) = generate_dataset(male=None)

    tb_contrastive = SummaryWriter(log_dir="runs3/contrastive")
    tb_regression = SummaryWriter(log_dir="runs3/regression")

    USE_CONTRASTIVE_TRAIN = True
    CONTRASTIVE_CKPT = "best_contrastive.pt"

    contrastive_model = MoCo().to(device)

    if USE_CONTRASTIVE_TRAIN:
        opt_c = torch.optim.AdamW(contrastive_model.parameters(), lr=1e-3, weight_decay=5e-4)
        sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=50)

        contrastive_model = train_contrastive(
            contrastive_model, opt_c,
            train_contrastive_loader, val_contrastive_loader, device,
            epochs=50, patience=15, ckpt=CONTRASTIVE_CKPT,
            scheduler=sch_c, writer=tb_contrastive
        )
    else:
        try:
            state = torch.load(CONTRASTIVE_CKPT, map_location=device)
            missing, unexpected = contrastive_model.load_state_dict(state, strict=False)
            print(f"[OK] Loaded contrastive checkpoint: {CONTRASTIVE_CKPT}")
            if missing or unexpected:
                print("[load] missing keys:", missing)
                print("[load] unexpected keys:", unexpected)
        except Exception as e:
            print(f"[WARN] Could not load {CONTRASTIVE_CKPT}: {e}")
            print("[Info] Proceeding with ImageNet-initialized ResNet-34 encoder.")

    reg_model = BoneAgePredictionModel(contrastive_model.encoder_q).to(device)
    print("[Sanity] reg in_features = 512 + gender(1) =", 512 + 1)

    enc_params, head_params = [], []
    for n, p in reg_model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith("contrastive_model.") else head_params).append(p)

    optimizer_r = torch.optim.AdamW(
        [
            {"params": enc_params,  "lr": 3e-4, "weight_decay": 5e-4},
            {"params": head_params, "lr": 2e-3, "weight_decay": 1e-4},
        ]
    )
    print("[Sanity] #params in optimizer:",
          sum(p.numel() for g in optimizer_r.param_groups for p in g['params']))

    scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=100)

    reg_model = train_bone_age_model(
        reg_model, optimizer_r, train_regression_loader, val_regression_loader, device,
        epochs=100, patience=25, ckpt='best_regression.pt',
        writer=tb_regression, scheduler=scheduler_r
    )

    tb_contrastive.close()
    tb_regression.close()
