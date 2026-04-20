# CTrain.py (drop-in)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from FixedHoughDataset import generate_dataset, SIGMA,MU

def print_backbone_trainable_layers(model):
    print("\n=== Backbone Layer Trainable Status ===")
    for name, param in model.contrastive_model.backbone.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"{name:50s} | {status}")
    print("======================================\n")


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
    def __init__(self, dim=128, K=65536, m=0.999, T=0.1):
        super().__init__()
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = Encoder(proj_dim=dim)
        self.encoder_k = Encoder(proj_dim=dim)

        # initialize key encoder parameters from query
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # queue buffer stored as (dim, K)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # must be called under @torch.no_grad()
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # k = m*k + (1-m)*q
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1. - self.m)

    # must be called under @torch.no_grad()
    @torch.no_grad()
    def _enqueue(self, keys):
        """
        keys: tensor [B, dim] (already detached). Will write into self.queue (dim, K).
        Wrap-around safe.
        """
        keys = keys.detach()
        B = keys.shape[0]
        ptr = int(self.queue_ptr)

        # keys -> transpose to (dim, B) when writing into (dim, K)
        if B >= self.K:
            # If batch >= queue, keep only last K keys
            keys = keys[-self.K:]
            B = keys.shape[0]
            ptr = 0
            self.queue[:, :] = keys.T
            self.queue_ptr[0] = (ptr + B) % self.K
            return

        if ptr + B <= self.K:
            self.queue[:, ptr:ptr + B] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :B - first] = keys[first:].T

        self.queue_ptr[0] = (ptr + B) % self.K

    @torch.no_grad()
    def monitor_queue(self, sample_size=2048):
        """Return dict of stats (safe — uses detach)."""
        q = self.queue.detach()  # (dim, K)
        norms = q.norm(dim=0)    # norms across dim for each column (K)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()

        # sample subset of columns for duplicate detection to avoid huge K*K
        K = q.shape[1]
        if K > sample_size:
            idx = torch.randint(0, K, (sample_size,), device=q.device)
            samp = q[:, idx].T  # (S, dim)
        else:
            samp = q.T  # (K, dim)

        sims = samp @ samp.T  # (S,S)
        sims.fill_diagonal_(0)
        duplicates_pct = float((sims > 0.999).float().mean().item()) * 100.0

        return {
            "ptr": int(self.queue_ptr.item()),
            "K": self.K,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            "duplicates_pct": duplicates_pct
        }

    def forward(self, im_q, im_k):
        """
        Pure forward: compute loss and return (loss, k).
        Must NOT mutate queue or encoder_k.
        """
        # 1) query features (with grad)
        q = self.encoder_q(im_q)                # [B, dim]
        q = F.normalize(q, dim=1)

        # 2) momentum update of key encoder BEFORE computing k (official order)
        #    NOTE: we *do the update* here only to compute k consistently for this forward pass,
        #    but we DO NOT enqueue or mutate queue here. The actual enqueue should happen
        #    after optimizer.step() to avoid autograd/version issues.
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)           # [B, dim]
            k = F.normalize(k, dim=1)

        # 3) compute logits using detached queue (avoid any autograd link)
        queue = self.queue.detach()             # (dim, K)
        # positive logits: q·k -> [B,1]
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(1)
        # negative logits: q·queue -> [B,K]
        l_neg = torch.einsum('nc,ck->nk', q, queue)

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss, k

# Replace your train_contrastive function with this version
def train_contrastive(model, optimizer, train_loader, val_loader, device,
                      epochs=200, ckpt='best_contrastive.pt',
                      writer: SummaryWriter = None, scheduler=None):
    best_val = float('inf')
    global_step = 0

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        train_losses = []
        progress = tqdm(total=len(train_loader), desc=f"E{epoch} Train", ncols=120)

        for it, batch in enumerate(train_loader):
            x1 = batch['images'].to(device, non_blocking=True)
            x2 = batch['images2'].to(device, non_blocking=True)

            # pure forward -> returns loss and keys (k)
            loss, k = model(x1, x2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ENQUEUE AFTER STEP (must be no_grad)
            with torch.no_grad():
                # 'k' is detached in model.forward's with torch.no_grad block, but ensure no grads:
                model._enqueue(k)

            train_losses.append(loss.item())

            progress.set_description(f"Train E{epoch:03d} | L={np.mean(train_losses):.4f}")
            progress.update(1)

            if writer:
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
            global_step += 1

        progress.close()

        # QUEUE MONITOR
        stats = model.monitor_queue()
        print(
            f"Queue ptr:{stats['ptr']} | mean_norm:{stats['mean_norm']:.4f} "
            f"| std_norm:{stats['std_norm']:.4f} | dup%:{stats['duplicates_pct']:.3f}"
        )
        if writer:
            writer.add_scalar("queue/mean_norm", stats['mean_norm'], epoch)
            writer.add_scalar("queue/std_norm", stats['std_norm'], epoch)
            writer.add_scalar("queue/duplicates_pct", stats['duplicates_pct'], epoch)

        train_loss = float(np.mean(train_losses))
        print(f"[Train] Epoch {epoch} Loss: {train_loss:.4f}")

        # VALIDATION (no enqueue)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x1 = batch['images'].to(device, non_blocking=True)
                x2 = batch['images2'].to(device, non_blocking=True)
                # forward returns loss,k but we discard k and do not enqueue
                loss_val, _ = model(x1, x2)
                val_losses.append(loss_val.item())
        val_loss = float(np.mean(val_losses))
        print(f"[Val] Epoch {epoch} Loss: {val_loss:.4f}")

        # checkpointing (you used val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
            print(f"[Checkpoint] Saved best model (val={best_val:.4f})")

        if writer:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            for gi, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group_{gi}", pg['lr'], epoch)

        if scheduler:
            scheduler.step()

    # load best
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[Done] Loaded best model (val={best_val:.4f})")
    return model



# ================= Regression model (plain deeper head) =================
class BoneAgePredictionModel(nn.Module):
    def __init__(self, contrastive_model,num_bins=20 ,num_classes=1):
        super().__init__()
        self.contrastive_model = contrastive_model
        self.pre_head_norm = nn.LayerNorm(512, eps=1e-6)
        # 👇 gender embedding
        self.gender_embed = nn.Embedding(num_embeddings=2, embedding_dim=8)
        in_dim = 512 + 8
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.coarse_head = nn.Linear(128, num_bins)
        self.fine_head = nn.Linear(128, 1)

        self.num_bins = num_bins
        self.register_buffer("bin_centers", torch.linspace(0, 228, num_bins))

    def forward(self, x, gender):
        feats = self.contrastive_model.forward_features(x)
        feats = self.pre_head_norm(feats)

        g_emb = self.gender_embed(gender.long())
        x = torch.cat([feats, g_emb], dim=1)

        h = self.shared(x)

        coarse_logits = self.coarse_head(h)
        fine_offset = self.fine_head(h)

        return coarse_logits, fine_offset


# ================= Eval: true item-wise MAE (months) =================
@torch.no_grad()
def eval_regression(model, loader, device, tta=False):
    model.eval()
    total_abs = 0.0
    n = 0

    def decode(model, coarse, offset):
        probs = torch.softmax(coarse, dim=1)
        bin_centers = model.bin_centers.to(probs.device).unsqueeze(0)
        bin_center = (probs * bin_centers).sum(dim=1)
        return bin_center + offset.squeeze(1)

    for batch in loader:
        x = batch["images"].to(device)
        y_norm = batch["labels"].to(device)
        g = batch["gender"].to(device)

        y_months = (y_norm * SIGMA + MU).squeeze(1)

        if not tta:
            coarse, offset = model(x, g)
            pred = decode(model, coarse, offset)

        else:
            coarse1, offset1 = model(x, g)
            coarse2, offset2 = model(torch.flip(x, dims=[3]), g)

            pred1 = decode(model, coarse1, offset1)
            pred2 = decode(model, coarse2, offset2)

            pred = 0.5 * (pred1 + pred2)

        total_abs += (pred - y_months).abs().sum().item()
        n += y_months.numel()

    return total_abs / n

@torch.no_grad()
def eval_test_regression(model, loader, device, tta=False):
    model.eval()
    total_abs = 0.0
    n = 0

    def decode(model, coarse, offset):
        probs = torch.softmax(coarse, dim=1)
        bin_centers = model.bin_centers.to(probs.device).unsqueeze(0)
        bin_center = (probs * bin_centers).sum(dim=1)
        return bin_center + offset.squeeze(1)

    for batch in loader:
        x = batch["images"].to(device)
        y_norm = batch["labels"].to(device)
        g = batch["gender"].to(device)

        y_months = (y_norm * SIGMA + MU).squeeze(1)

        if not tta:
            coarse, offset = model(x, g)
            pred = decode(model, coarse, offset)

        else:
            x_list = [x, torch.flip(x, dims=[3])]
            preds = []

            for x_aug in x_list:
                coarse, offset = model(x_aug, g)
                preds.append(decode(model, coarse, offset))

            pred = torch.stack(preds).mean(0)

        total_abs += (pred - y_months).abs().sum().item()
        n += y_months.numel()

    return total_abs / n


# ================= Train regression (Huber + EMA + Cosine) =================
def train_bone_age_model(model, optimizer, train_loader, val_loader,test_loader, device,
                         epochs=50, patience=20, ckpt='best_regression.pt',
                         writer=None, scheduler=None):



    stopper = EarlyStopping(patience=patience, mode='min', checkpoint_path=ckpt)
    global_step = 0

    def create_ema(model, device):
        from copy import deepcopy
        ema = deepcopy(model).to(device)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    model_ema = create_ema(model, device)

    EMA_START_EPOCH = 1

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

        # ============================================================
        # 🔓 PROGRESSIVE UNFREEZING WITH OPTIMIZER + SCHEDULER RESET
        # ============================================================

        if epoch == 5:
            print("[Info] Unfreezing layer4 (highest-level features)")

            for p in model.backbone[7].parameters():
                p.requires_grad = True

            print_backbone_trainable_layers(model)

            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone[7].parameters(), "lr": 5e-5},

                    # 🔥 NEW HEAD PARAMS
                    {"params": model.shared.parameters(), "lr": 1e-3},
                    {"params": model.coarse_head.parameters(), "lr": 1e-3},
                    {"params": model.fine_head.parameters(), "lr": 1e-3},
                ],
                weight_decay=1e-5,
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - epoch,
            )

            model_ema = create_ema(model, device)

            print("[Info] Optimizer & scheduler reset after unfreezing layer4")


        elif epoch == 15:
            print("[Info] Unfreezing layer3 (mid-level features)")

            for p in model.backbone[6].parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone[6].parameters(), "lr": 1e-5},
                    {"params": model.backbone[7].parameters(), "lr": 5e-5},

                    # 🔥 NEW HEAD PARAMS
                    {"params": model.shared.parameters(), "lr": 5e-4},
                    {"params": model.coarse_head.parameters(), "lr": 5e-4},
                    {"params": model.fine_head.parameters(), "lr": 5e-4},
                ],
                weight_decay=1e-5,
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - epoch,
            )
            model_ema = create_ema(model, device)


            print("[Info] Optimizer & scheduler reset after unfreezing layer3")

        model.train()
        train_mae_months_hist = []
        progress = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch}", ncols=110)

        if epoch < 10:
            ema_decay = 0.99
        elif epoch < 25:
            ema_decay = 0.999
        else:
            ema_decay = 0.9997

        for it, batch in enumerate(train_loader):
            x = batch['images'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            g = batch['gender'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # forward
            pred_coarse, pred_offset = model(x, g)

            # =========================
            # 🔥 to months
            # =========================
            y_months = (y * SIGMA + MU).view(-1, 1)

            # =========================
            # 🔥 bin setup
            # =========================
            bin_centers = model.bin_centers.to(y_months.device).unsqueeze(0)
            bin_width = 228 / model.num_bins
            sigma_bins = bin_width * 0.5

            # =========================
            # 🔥 SOFT COARSE TARGETS
            # =========================
            dist = y_months - bin_centers  # [B, num_bins]

            soft_targets = torch.exp(-0.5 * (dist / sigma_bins) ** 2)
            soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

            # =========================
            # 🔥 COARSE LOSS
            # =========================
            log_probs = torch.log_softmax(pred_coarse, dim=1)
            probs = log_probs.exp()

            loss_coarse = -(soft_targets * log_probs).sum(dim=1).mean()

            # =========================
            # 🔥 DIFFERENTIABLE BIN CENTER
            # =========================
            bin_centers = model.bin_centers.to(probs.device)
            bin_center = (probs * bin_centers.unsqueeze(0)).sum(dim=1)

            pred_age = bin_center + pred_offset

            # =========================
            # 🔥 FINE LOSS
            # =========================
            loss_fine = F.smooth_l1_loss(pred_age, y_months, beta=1.5)

            # =========================
            # 🔥 TOTAL LOSS
            # =========================
            loss = 0.7 * loss_coarse + 0.3 * loss_fine

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch >= EMA_START_EPOCH:
                ema_update(model, model_ema, decay=ema_decay)

            with torch.no_grad():
                mae_months_batch = (pred_age - y_months).abs().mean().item()
            train_mae_months_hist.append(mae_months_batch)

            progress.set_description(
                desc=f"Train - Ep:{epoch:03d} | It:{it:04d} | MAE(mo): {np.mean(train_mae_months_hist):.2f}"
            )
            progress.update(1)

            if writer is not None:
                writer.add_scalar("train/batch_mae_months", mae_months_batch, global_step)
                writer.add_scalar("train/loss_total", loss.item(), global_step)
                writer.add_scalar("train/loss_coarse", loss_coarse.item(), global_step)
                writer.add_scalar("train/loss_fine", loss_fine.item(), global_step)
            global_step += 1

        progress.close()

        train_mae_months = float(np.mean(train_mae_months_hist))
        tqdm.write(f"Train - Ep:{epoch} | MAE (months): {train_mae_months:.2f}")
        if writer is not None:
            writer.add_scalar("epoch/train_mae_months", train_mae_months, epoch)

        val_raw_no_tta = eval_regression(model, val_loader, device, tta=False)
        val_raw_tta = eval_regression(model, val_loader, device, tta=True)

        val_ema_no_tta = eval_regression(model_ema, val_loader, device, tta=False)
        val_ema_tta = eval_regression(model_ema, val_loader, device, tta=True)
        if writer is not None:
            writer.add_scalar("epoch/val_mae_months", val_ema_tta, epoch)

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
    # -------------------------
    # FINAL TEST EVALUATION
    # -------------------------
    test_mae = eval_test_regression(
        model_ema,  # ALWAYS EMA
        test_loader,
        device,
        tta=True
    )

    print(f"[TEST] MAE (months): {test_mae:.2f}")

    if writer is not None:
        writer.add_scalar("test/mae_months", test_mae, 0)

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
        test_regression_ds,
        train_contrastive_loader,
        val_contrastive_loader,
        train_regression_loader,
        val_regression_loader,test_regression_loader
    ) = generate_dataset(male=None)

    tb_contrastive = SummaryWriter(log_dir="runs3/contrastive")
    tb_regression = SummaryWriter(log_dir="runs3/regression")

    USE_CONTRASTIVE_TRAIN = False
    CONTRASTIVE_CKPT = "best_contrastive.pt"

    contrastive_model = MoCo().to(device)

    if USE_CONTRASTIVE_TRAIN:
        opt_c = torch.optim.SGD(contrastive_model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
        sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=100)

        contrastive_model = train_contrastive(
            contrastive_model, opt_c,
            train_contrastive_loader, val_contrastive_loader, device,
            epochs=100, ckpt=CONTRASTIVE_CKPT,
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

    # 🔒 FREEZE ENTIRE CONTRASTIVE ENCODER (backbone + projector)
    for p in reg_model.contrastive_model.parameters():
        p.requires_grad = False

    reg_model.contrastive_model.projector = nn.Identity()

    print_backbone_trainable_layers(reg_model)

    optimizer_r = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, reg_model.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )
    scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=100)
    for i, layer in enumerate(reg_model.contrastive_model.backbone):
        print(i, layer)
    reg_model = train_bone_age_model(
        reg_model, optimizer_r, train_regression_loader, val_regression_loader,test_regression_loader, device,
        epochs=100, patience=25, ckpt='best_regression.pt',
        writer=tb_regression, scheduler=scheduler_r
    )


    tb_contrastive.close()
    tb_regression.close()

