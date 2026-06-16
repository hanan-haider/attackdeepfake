"""
trainer.py
==========
Generic training wrapper for the Attack Agnostic Dataset pipeline.
All 4 stability fixes from lcnn_stabilized.py integrated inline.

Changes vs original:
  CHANGE 1  SmoothedBCEWithLogitsLoss replaces BCEWithLogitsLoss  (FIX 7)
  CHANGE 2  pos_weight computed per fold from training split       (FIX 2)
  CHANGE 3  build_scheduler (warmup+cosine) replaces OneCycleLR   (FIX 3)
  CHANGE 4  Gradient clipping (max_norm=5.0) added to train loop
  CHANGE 5  criterion used for eval loss (consistent with train)
  CHANGE 6  scheduler.step() called per EPOCH not per batch
  KEPT      All original interfaces, DataLoader config, logging
  KEPT      test_dataset / test_len split logic
  KEPT      best_model tracking by test accuracy
  KEPT      num_workers=4, pin_memory=True
"""

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfadetect import cnn_features
from dfadetect.models.lcnn import (
    SmoothedBCEWithLogitsLoss,
    compute_pos_weight,
    build_scheduler,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOGGER = logging.getLogger(__name__)


# ======================================================================
#  Data setting dataclass — unchanged from original
# ======================================================================

@dataclass
class NNDataSetting:
    use_cnn_features: bool


# ======================================================================
#  Base Trainer — unchanged from original
# ======================================================================

class Trainer:
    """
    Lightweight wrapper for gradient-descent training.
    Stores training configuration and epoch-level test losses.

    Args:
        epochs           (int):      Number of training epochs.
        batch_size       (int):      Samples per batch.
        device           (str):      'cpu' or 'cuda'.
        optimizer_fn     (Callable): Optimizer class (default Adam).
        optimizer_kwargs (dict):     Kwargs passed to optimizer_fn.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = None,
    ) -> None:
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.device           = device
        self.optimizer_fn     = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None \
                                else {"lr": 1e-3}
        self.epoch_test_losses: List[float] = []


# ======================================================================
#  forward_and_loss — updated signature for pos_weight support
# ======================================================================

def forward_and_loss(
    model: torch.nn.Module,
    criterion: SmoothedBCEWithLogitsLoss,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    pos_weight: torch.Tensor = None,
    **kwargs,
):
    """
    Single forward pass + loss computation.

    Args:
        model      : LCNN model.
        criterion  : SmoothedBCEWithLogitsLoss instance.
        batch_x    : Feature tensor [B, C, F, T].
        batch_y    : Label tensor [B, 1] float.
        pos_weight : Per-fold imbalance weight [1] (FIX 2). Optional.

    Returns:
        (batch_out [B,1], batch_loss scalar)
    """
    batch_out  = model(batch_x)
    batch_loss = criterion(batch_out, batch_y, pos_weight=pos_weight)
    return batch_out, batch_loss


# ======================================================================
#  GDTrainer — full rewrite with all 4 integration fixes
# ======================================================================

class GDTrainer(Trainer):
    """
    Gradient-descent trainer with all EER stabilization fixes applied.

    Key changes vs original:
      - SmoothedBCEWithLogitsLoss (label smoothing 0.05)      FIX 7
      - pos_weight = n_fake/n_real per fold                   FIX 2
      - LR warmup 5 epochs + CosineAnnealingWarmRestarts      FIX 3
      - Gradient clipping max_norm=5.0                        additional
      - scheduler.step() per epoch                            FIX 3
      - Eval uses same criterion as train (consistent losses)
    """

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        nn_data_setting: NNDataSetting,
        cnn_features_setting: cnn_features.CNNFeaturesSetting,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        logging_prefix: str = "",
        pos_weight: Optional[torch.FloatTensor] = None,   # kept for API compat
    ):
        """
        Train model on dataset with full stability fixes.

        Args:
            dataset              : Training dataset (or full dataset if
                                   test_dataset is None).
            model                : LCNN model instance.
            nn_data_setting      : NNDataSetting(use_cnn_features=True/False).
            cnn_features_setting : CNNFeaturesSetting for LFCC/MFCC.
            test_len             : Fraction of dataset for test split
                                   (used only if test_dataset is None).
            test_dataset         : Explicit test dataset (preferred).
            logging_prefix       : String prefix for log messages.
            pos_weight           : Ignored — computed automatically from
                                   training split (FIX 2). Kept for API
                                   backward compatibility.

        Returns:
            model with best weights loaded (by test accuracy).
        """

        # ── Dataset split ─────────────────────────────────────────────
        if test_dataset is not None:
            train = dataset
            test  = test_dataset
        else:
            test_len  = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            train, test = torch.utils.data.random_split(
                dataset, [train_len, test_len]
            )

        # ── DataLoaders ───────────────────────────────────────────────
        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

        # ── CHANGE 1: SmoothedBCEWithLogitsLoss replaces BCEWithLogitsLoss
        criterion = SmoothedBCEWithLogitsLoss(smoothing=0.05)

        # ── CHANGE 2: pos_weight computed from training split per fold
        # Computes n_fake / n_real on the actual training data.
        # This corrects the per-fold class imbalance in AAD.
        LOGGER.info("Computing per-fold pos_weight from training split...")
        fold_pos_weight = compute_pos_weight(train, self.device)
        LOGGER.info(f"pos_weight = {fold_pos_weight.item():.4f} "
                    f"(n_fake/n_real on training split)")

        # ── Optimizer ─────────────────────────────────────────────────
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        # ── CHANGE 3: Warmup + CosineAnnealingWarmRestarts
        # Replaces OneCycleLR. Called per EPOCH not per batch.
        # 5-epoch linear warmup prevents fold-dependent saddle points.
        scheduler = build_scheduler(
            optim,
            warmup_epochs=5,
            cosine_T0=10,
            cosine_T_mult=2,
            eta_min=1e-6,
        )

        best_model = None
        best_acc   = 0.0

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            # ── TRAINING ──────────────────────────────────────────────
            running_loss = 0.0
            num_correct  = 0.0
            num_total    = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(self.device)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x,
                        cnn_features_setting=cnn_features_setting,
                    )

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                # CHANGE 2: pass fold_pos_weight to criterion
                batch_out, batch_loss = forward_and_loss(
                    model,
                    criterion,
                    batch_x,
                    batch_y,
                    pos_weight=fold_pos_weight,
                    use_cuda=use_cuda,
                )

                batch_pred   = (torch.sigmoid(batch_out) + 0.5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
                running_loss += batch_loss.item() * batch_size

                if i % 100 == 0:
                    LOGGER.info(
                        f"[Epoch {epoch:04d}] [Step {i:05d}] "
                        f"| Loss: {running_loss / num_total:.4f} "
                        f"| Acc: {num_correct / num_total * 100:.2f}%"
                    )

                optim.zero_grad()
                batch_loss.backward()

                # CHANGE 4: Gradient clipping — prevents exploding
                # gradients in BLSTM layers across folds
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )

                optim.step()
                # NOTE: scheduler.step() is called PER EPOCH below (FIX 3)
                # Do NOT call scheduler.step() here inside the batch loop.

            running_loss   /= num_total
            train_accuracy  = (num_correct / num_total) * 100

            print(
            f"\nEpoch [{epoch+1}/{self.epochs}]: "
            f"train/{logging_prefix}__loss: {running_loss:.4f}, "
            f"train/{logging_prefix}__accuracy: {train_accuracy:.2f}%"
            )
            torch.cuda.empty_cache()

            # ── CHANGE 3: scheduler step PER EPOCH ────────────────────
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            LOGGER.info(f"LR after epoch {epoch+1}: {current_lr}")

            # ── EVALUATION ────────────────────────────────────────────
            test_running_loss = 0.0
            num_correct       = 0.0
            num_total         = 0.0
            model.eval()

            with torch.no_grad():
                for batch_x, _, batch_y in test_loader:
                    batch_size = batch_x.size(0)
                    num_total += batch_size

                    batch_x = batch_x.to(self.device)

                    if nn_data_setting.use_cnn_features:
                        batch_x = cnn_features.prepare_feature_vector(
                            batch_x,
                            cnn_features_setting=cnn_features_setting,
                        )

                    batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                    batch_out = model(batch_x)

                    # CHANGE 5: use same criterion as train (consistent eval)
                    # pos_weight=None during eval — no imbalance correction
                    # needed for loss reporting, only for gradient updates.
                    batch_loss = criterion(batch_out, batch_y, pos_weight=None)

                    test_running_loss += batch_loss.item() * batch_size

                    batch_pred   = (torch.sigmoid(batch_out) + 0.5).int()
                    num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            if num_total == 0:
                num_total = 1

            test_running_loss /= num_total
            test_acc           = 100.0 * (num_correct / num_total)

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/{logging_prefix}__loss: {test_running_loss:.4f}, "
                f"test/{logging_prefix}__accuracy: {test_acc:.2f}%"
            )

            self.epoch_test_losses.append(test_running_loss)

            if best_model is None or test_acc > best_acc:
                best_acc   = test_acc
                best_model = deepcopy(model.state_dict())
                LOGGER.info(
                    f"  → New best model saved (test_acc={best_acc:.2f}%)"
                )

            LOGGER.info(
                f"[{epoch:04d}]: loss={running_loss:.4f} "
                f"| train_acc={train_accuracy:.2f}% "
                f"| test_acc={test_acc:.2f}% "
                f"| best_acc={best_acc:.2f}%"
            )

        model.load_state_dict(best_model)
        return model