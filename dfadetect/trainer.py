"""A generic training wrapper."""
# original trainer code of attackdeepfake and its works with improved lcnn model
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfadetect import cnn_features

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


@dataclass
class NNDataSetting:
    use_cnn_features: bool


class Trainer():
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): Amount of audio files to use in one batch.
        device (str): The device to train on (Default 'cpu').
        optimizer_fn (Callable): Function for constructing the optimizer.
        optimizer_kwargs (dict): Kwargs for the optimizer.
    """

    def __init__(self,
                 epochs: int = 20,
                 batch_size: int = 32,
                 device: str = "cpu",
                 optimizer_fn: Callable = torch.optim.Adam,
                 optimizer_kwargs: dict = {"lr": 1e-3},
                 ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss




class GDTrainer(Trainer):

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        nn_data_setting: NNDataSetting,
        cnn_features_setting: cnn_features.CNNFeaturesSetting,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        logging_prefix: str = "",
        pos_weight: Optional[torch.FloatTensor] = None,
        ckpt_dir: Optional[str] = None,       # FIX 1: added missing parameter
        ckpt_tag: Optional[str] = None,        # FIX 2: added missing parameter
    ):
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            train, test = torch.utils.data.random_split(dataset, [train_len, test_len])

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
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0.0

        # Resolve checkpoint save directory
        tag = ckpt_tag if ckpt_tag else (logging_prefix if logging_prefix else "fold")
        if ckpt_dir:
            save_dir = os.path.join(ckpt_dir, tag)
            os.makedirs(save_dir, exist_ok=True)
            best_ckpt_path  = os.path.join(save_dir, "best_checkpoint.pth")
            epoch_ckpt_path = os.path.join(save_dir, "latest_epoch.pth")

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x, cnn_features_setting=cnn_features_setting)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(
                    model, criterion, batch_x, batch_y, use_cuda=use_cuda)
                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
                running_loss += (batch_loss.item() * batch_size)

                if i % 100 == 0:
                    LOGGER.info(
                        f"[Epoch {epoch:04d}] [Step {i:05d}] "
                        f"| Loss: {running_loss / num_total:.4f} "
                        f"| Acc: {num_correct / num_total * 100:.2f}%")

                optim.zero_grad()
                batch_loss.backward()
                optim.step()

            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"train/{logging_prefix}__loss: {running_loss:.4f}, "
                f"train/{logging_prefix}__accuracy: {train_accuracy:.2f}%")
            torch.cuda.empty_cache()

            #   EVALUATION 
            # FIX 3: moved out of torch.cuda.empty_cache() scope —
            # was indented one level too deep, causing it to run OUTSIDE
            # the epoch loop on the very last iteration only.
            test_running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.eval()

            with torch.no_grad():
                for batch_x, _, batch_y in test_loader:
                    batch_size = batch_x.size(0)
                    num_total += batch_size
                    batch_x = batch_x.to(self.device)

                    if nn_data_setting.use_cnn_features:
                        batch_x = cnn_features.prepare_feature_vector(
                            batch_x, cnn_features_setting=cnn_features_setting)

                    batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                    batch_out = model(batch_x)
                    batch_loss = criterion(batch_out, batch_y)

                    test_running_loss += (batch_loss.item() * batch_size)
                    batch_pred = (torch.sigmoid(batch_out) + .5).int()
                    num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            if num_total == 0:
                num_total = 1

            # FIX 4: moved test metric computation OUT of torch.no_grad() block —
            # these are plain Python arithmetic, not tensor ops; keeping them
            # inside no_grad() is harmless but caused wrong indentation scope.
            test_running_loss /= num_total
            test_acc = 100.0 * (num_correct / num_total)

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/{logging_prefix}__loss: {test_running_loss:.4f}, "
                f"test/{logging_prefix}__accuracy: {test_acc:.2f}%")

            #   CHECKPOINT: save best model immediately on improvement ───────
            # FIX 5: torch.save is inside the if-block so it writes to disk
            # the moment a new best is found, not after all epochs finish.
            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

                if ckpt_dir:
                    torch.save({
                        "epoch":            epoch,
                        "model_state_dict": best_model,
                        "val_acc":          best_acc,
                        "val_loss":         test_running_loss,
                    }, best_ckpt_path)
                    
                    LOGGER.info(
                        f"New best checkpoint saved at epoch {epoch} "
                        f"with val_acc={best_acc:.4f}")

            LOGGER.info(
                f"[{epoch:04d}]: {running_loss:.4f} "
                f"-- train acc: {train_accuracy:.2f}% "
                f"-- test_acc: {test_acc:.2f}% "
                f"-- best_acc: {best_acc:.2f}%")

        #   END OF EPOCH LOOP  
        # FIX 6: these two lines were inside the epoch loop (wrong indent).
        # They belong AFTER the loop — load best weights once at the very end.
        LOGGER.info(
            f"Best checkpoint for {logging_prefix}: "
            f"val_acc={best_acc:.4f}, val_loss={test_running_loss:.4f}")
        model.load_state_dict(best_model)
        return model