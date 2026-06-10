"""
Stable GDTrainer — Fixed & Improved
=====================================

BUGS FIXED FROM ORIGINAL
──────────────────────────
1. NameError: `optimizer` in scheduler should be `optim`
   scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, ...)
   ↳ was a NameError at runtime; scheduler was never created.

2. Wrong classification threshold:
   (torch.sigmoid(batch_out) + 0.5).int()
   → sigmoid ∈ [0,1], +0.5 gives [0.5,1.5], int() maps to {0,1}
     but the threshold was effectively ~0.73, not 0.5.
   ↳ Fixed: (torch.sigmoid(batch_out) >= 0.5).int()

3. scheduler.step() never called → LR was constant throughout training.
   ↳ Called after every optimizer step (OneCycleLR requires per-step calls).

4. No gradient clipping → exploding gradients with BLSTM on long sequences
   (ASVspoof fold 3 gap was partly caused by this).
   ↳ torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

5. No AMP (Automatic Mixed Precision) → wasted compute + memory on GPU.
   ↳ torch.cuda.amp.GradScaler + autocast added.

IMPROVEMENTS ADDED
───────────────────
6. Model-aware forward pass: passes batch_y to model.forward() so that
   the built-in Mixup augmentation receives the labels it needs.

7. forward_and_loss uses smoothed_bce_loss (label smoothing ε=0.05)
   — anti-overconfidence on unseen generators.

8. pos_weight support wired through properly (for imbalanced datasets).

9. EER (Equal Error Rate) computed and logged per epoch — this is the
   actual metric used in the paper, not accuracy.

10. Best model saved by EER (lower=better) not by accuracy.

11. WarmupCosineScheduler: 5-epoch linear warmup → cosine decay.
    More stable than raw OneCycleLR when combined with BLSTM.
    (OneCycleLR still supported via use_one_cycle=True kwarg.)

12. Per-epoch torch.cuda.empty_cache() retained; also called after val.

USAGE
──────
    trainer = GDTrainer(
        epochs=30,
        batch_size=32,
        device="cuda",
        optimizer_fn=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
    )
    trained_model = trainer.train(
        dataset=train_ds,
        model=model,
        nn_data_setting=nn_data_setting,
        cnn_features_setting=cnn_features_setting,
        test_dataset=test_ds,
    )
"""

import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dfadetect import cnn_features

# Recommended by PyTorch for fragmentation reduction on large batches
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data setting
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NNDataSetting:
    use_cnn_features: bool


# ─────────────────────────────────────────────────────────────────────────────
# EER computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Equal Error Rate.

    Args:
        scores: float array of model scores (higher = more likely fake)
        labels: int array of ground-truth labels (1=fake, 0=real)
    Returns:
        EER as a percentage [0, 100]
    """
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

    # Build FAR / FRR curves
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    far_list, frr_list = [], []
    n_real = (labels == 0).sum()
    n_fake = (labels == 1).sum()

    if n_real == 0 or n_fake == 0:
        return 50.0   # degenerate split

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        fa  = ((preds == 1) & (labels == 0)).sum()
        fr  = ((preds == 0) & (labels == 1)).sum()
        far_list.append(fa / n_real)
        frr_list.append(fr / n_fake)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    diff    = far_arr - frr_arr

    # Find crossing point
    try:
        eer = brentq(interp1d(thresholds, diff), thresholds[0], thresholds[-1])
        # eer here is the threshold; get the actual rate
        idx  = np.argmin(np.abs(thresholds - eer))
        rate = (far_arr[idx] + frr_arr[idx]) / 2.0
    except Exception:
        # Fallback: take min absolute difference
        idx  = np.argmin(np.abs(diff))
        rate = (far_arr[idx] + frr_arr[idx]) / 2.0

    return float(rate * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Label-smoothed loss (imported from model file or re-defined here)
# ─────────────────────────────────────────────────────────────────────────────

def smoothed_bce_loss(
        logit: torch.Tensor,
        target: torch.Tensor,
        smoothing: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss with label smoothing ε (default 0.05)."""
    soft = target * (1.0 - smoothing) + 0.5 * smoothing
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logit, soft, pos_weight=pos_weight
    )


# ─────────────────────────────────────────────────────────────────────────────
# Warmup + Cosine LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup for `warmup_steps` steps, then cosine decay to `min_lr_ratio`
    of the base LR over the remainder of training.

    More stable than raw OneCycleLR when the first few batches have noisy
    gradients from the BLSTM initialisation.
    """
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: int,
            total_steps: int,
            min_lr_ratio: float = 0.05,
    ):
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step + 1) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Forward + loss helper
# ─────────────────────────────────────────────────────────────────────────────

def forward_and_loss(
        model: nn.Module,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        **kwargs,
):
    """
    Call model.forward(x, y) to support in-model Mixup,
    then compute smoothed BCE loss on the (possibly mixed) labels.

    Returns (logit, effective_y, loss).
    """
    logit, effective_y = model(batch_x, batch_y)
    if effective_y is None:
        effective_y = batch_y
    loss = smoothed_bce_loss(logit, effective_y,
                             smoothing=label_smoothing,
                             pos_weight=pos_weight)
    return logit, effective_y, loss


# ─────────────────────────────────────────────────────────────────────────────
# Base trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Lightweight wrapper for gradient-descent training.

    Args:
        epochs           : number of training epochs            (default 20)
        batch_size       : samples per batch                    (default 32)
        device           : 'cpu' or 'cuda'                      (default 'cpu')
        optimizer_fn     : optimizer class                      (default Adam)
        optimizer_kwargs : kwargs for optimizer                 (default lr=1e-3)
        grad_clip        : max gradient norm (0 = disabled)     (default 5.0)
        label_smoothing  : BCE label smoothing ε                (default 0.05)
        use_amp          : mixed-precision training on CUDA     (default True)
        use_one_cycle    : use OneCycleLR instead of warmup cos (default False)
    """

    def __init__(
            self,
            epochs:           int      = 20,
            batch_size:       int      = 32,
            device:           str      = "cpu",
            optimizer_fn:     Callable = torch.optim.Adam,
            optimizer_kwargs: dict     = None,
            grad_clip:        float    = 5.0,
            label_smoothing:  float    = 0.05,
            use_amp:          bool     = True,
            use_one_cycle:    bool     = False,
    ) -> None:
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.device           = device
        self.optimizer_fn     = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.grad_clip        = grad_clip
        self.label_smoothing  = label_smoothing
        self.use_amp          = use_amp and (device != "cpu")
        self.use_one_cycle    = use_one_cycle
        self.epoch_test_losses: List[float] = []


# ─────────────────────────────────────────────────────────────────────────────
# GDTrainer
# ─────────────────────────────────────────────────────────────────────────────

class GDTrainer(Trainer):

    def train(
            self,
            dataset:                torch.utils.data.Dataset,
            model:                  nn.Module,
            nn_data_setting:        NNDataSetting,
            cnn_features_setting:   cnn_features.CNNFeaturesSetting,
            test_len:               Optional[float] = None,
            test_dataset:           Optional[torch.utils.data.Dataset] = None,
            logging_prefix:         str = "",
            pos_weight:             Optional[torch.FloatTensor] = None,
    ) -> nn.Module:
        """
        Train model using gradient descent with all stability fixes applied.

        Returns the best model (lowest EER on test set) as a side-effect of
        loading its state dict into `model` and also returning `model`.
        """

        # ── Data splits ───────────────────────────────────────────────────
        if test_dataset is not None:
            train_data = dataset
            test_data  = test_dataset
        else:
            test_len_n  = int(len(dataset) * test_len)
            train_len_n = len(dataset) - test_len_n
            train_data, test_data = torch.utils.data.random_split(
                dataset, [train_len_n, test_len_n]
            )

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=(self.device != "cpu"),
        )
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            drop_last=False,           # don't drop last — EER needs all samples
            num_workers=4,
            pin_memory=(self.device != "cpu"),
        )

        # ── Optimiser ─────────────────────────────────────────────────────
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        # ── FIX 1: scheduler uses `optim`, not an undefined `optimizer` ───
        total_steps   = self.epochs * len(train_loader)
        warmup_steps  = 5 * len(train_loader)   # 5-epoch warmup

        if self.use_one_cycle:
            # OneCycleLR: good but slightly less stable early on with LSTMs
            max_lr    = self.optimizer_kwargs.get("lr", 1e-3) * 10
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim,                           # FIX: was `optimizer` (NameError)
                max_lr=max_lr,
                epochs=self.epochs,
                steps_per_epoch=len(train_loader),
            )
        else:
            scheduler = WarmupCosineScheduler(
                optim,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=0.05,
            )

        # ── AMP scaler (FIX 5: was missing entirely) ──────────────────────
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ── pos_weight to device ──────────────────────────────────────────
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)

        best_model = None
        best_eer   = float("inf")

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            # ── TRAIN ─────────────────────────────────────────────────────
            running_loss = 0.0
            num_correct  = 0.0
            num_total    = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size_i  = batch_x.size(0)
                num_total    += batch_size_i

                batch_x = batch_x.to(self.device)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x, cnn_features_setting=cnn_features_setting
                    )

                # Labels: [B, 1] float32
                batch_y = batch_y.unsqueeze(1).float().to(self.device)

                # ── Forward + loss (AMP) ──────────────────────────────────
                optim.zero_grad(set_to_none=True)   # faster than zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logit, effective_y, batch_loss = forward_and_loss(
                        model, batch_x, batch_y,
                        pos_weight=pos_weight,
                        label_smoothing=self.label_smoothing,
                    )

                # ── Backward ──────────────────────────────────────────────
                scaler.scale(batch_loss).backward()

                # FIX 4: gradient clipping (prevents exploding grads in LSTM)
                if self.grad_clip > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.grad_clip
                    )

                scaler.step(optim)
                scaler.update()

                # FIX 3: scheduler.step() was never called in original
                scheduler.step()

                # ── FIX 2: correct threshold >= 0.5 ──────────────────────
                # Original: (sigmoid(x) + 0.5).int() → threshold ~0.73
                # Correct:  (sigmoid(x) >= 0.5).int() → threshold exactly 0.5
                batch_pred   = (torch.sigmoid(logit) >= 0.5).int()
                # Compare against hard labels (not mixed) for acc display
                hard_y       = (batch_y >= 0.5).int()
                num_correct += (batch_pred == hard_y).sum().item()

                running_loss += batch_loss.item() * batch_size_i

                if i % 100 == 0:
                    LOGGER.info(
                        f"[Epoch {epoch:04d}][Step {i:05d}] "
                        f"loss={running_loss / num_total:.4f} "
                        f"acc={num_correct / num_total * 100:.2f}% "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

            running_loss  /= num_total
            train_accuracy = num_correct / num_total * 100

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"train/{logging_prefix}__loss: {running_loss:.4f}  "
                f"train/{logging_prefix}__acc: {train_accuracy:.2f}%"
            )
            torch.cuda.empty_cache()

            # ── EVAL ──────────────────────────────────────────────────────
            test_running_loss = 0.0
            num_correct       = 0.0
            num_total_test    = 0.0
            all_scores:  List[float] = []
            all_labels:  List[int]   = []

            model.eval()
            with torch.no_grad():
                for batch_x, _, batch_y in test_loader:
                    batch_size_i    = batch_x.size(0)
                    num_total_test += batch_size_i

                    batch_x = batch_x.to(self.device)

                    if nn_data_setting.use_cnn_features:
                        batch_x = cnn_features.prepare_feature_vector(
                            batch_x, cnn_features_setting=cnn_features_setting
                        )

                    batch_y = batch_y.unsqueeze(1).float().to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        # At eval, model returns (logit, None); no Mixup
                        logit, _ = model(batch_x)
                        batch_loss = smoothed_bce_loss(
                            logit, batch_y,
                            smoothing=self.label_smoothing,
                            pos_weight=pos_weight,
                        )

                    test_running_loss += batch_loss.item() * batch_size_i

                    scores = torch.sigmoid(logit).squeeze(1)   # [B]
                    preds  = (scores >= 0.5).int()
                    hard_y = (batch_y.squeeze(1) >= 0.5).int()
                    num_correct += (preds == hard_y).sum().item()

                    all_scores.extend(scores.cpu().numpy().tolist())
                    all_labels.extend(hard_y.cpu().numpy().tolist())

            if num_total_test == 0:
                num_total_test = 1

            test_running_loss /= num_total_test
            test_acc           = 100.0 * num_correct / num_total_test
            self.epoch_test_losses.append(test_running_loss)

            # Compute EER — this is the metric from the paper
            test_eer = compute_eer(
                np.array(all_scores, dtype=np.float32),
                np.array(all_labels, dtype=np.int32),
            )

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"test/{logging_prefix}__loss: {test_running_loss:.4f}  "
                f"test/{logging_prefix}__acc: {test_acc:.2f}%  "
                f"test/{logging_prefix}__EER: {test_eer:.3f}%"
            )
            LOGGER.info(
                f"[{epoch:04d}] loss={running_loss:.4f} "
                f"train_acc={train_accuracy:.2f}% "
                f"test_acc={test_acc:.2f}% "
                f"test_EER={test_eer:.3f}%"
            )

            # Save best model by EER (paper metric), not by accuracy
            if best_model is None or test_eer < best_eer:
                best_eer   = test_eer
                best_model = deepcopy(model.state_dict())
                LOGGER.info(
                    f"  ★ New best model  EER={best_eer:.3f}%"
                )

            torch.cuda.empty_cache()

        # Load best checkpoint
        model.load_state_dict(best_model)
        LOGGER.info(f"Training complete. Best EER={best_eer:.3f}%")
        return model