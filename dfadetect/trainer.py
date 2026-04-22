"""A generic training wrapper."""
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfadetect import cnn_features
#from dfadetect.datasets import TransformDataset

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOGGER = logging.getLogger(__name__)

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
        batch_size (int): The amount of audio files to consider in one batch (Default: 32).
        optimizer_fn (Callable): Function for constructing the optimzer.
        optimizer_kwargs (dict): Kwargs for the optimzer.
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
    ):
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)

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
        best_acc = 0

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(self.device)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(batch_x, cnn_features_setting=cnn_features_setting)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(model, criterion, batch_x, batch_y, use_cuda=use_cuda)
                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                running_loss += (batch_loss.item() * batch_size)

                if i % 100 == 0:
                    LOGGER.info(
                         f"[Epoch {epoch:04d}] [Step {i:05d}] | Loss: {running_loss / num_total:.4f} | Acc: {num_correct / num_total * 100:.2f}%")

                optim.zero_grad()
                batch_loss.backward()
                optim.step()

            running_loss /= num_total
            train_accuracy = (num_correct/num_total)*100

            LOGGER.info(f"Epoch [{epoch+1}/{self.epochs}]: train/{logging_prefix}__loss: {running_loss}, train/{logging_prefix}__accuracy: {train_accuracy}")
            torch.cuda.empty_cache()

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
                        batch_x = cnn_features.prepare_feature_vector(batch_x, cnn_features_setting=cnn_features_setting)
    
                    batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                    batch_out = model(batch_x)
                    batch_loss = criterion(batch_out, batch_y)
    
                    test_running_loss += (batch_loss.item() * batch_size)
    
                    batch_pred = (torch.sigmoid(batch_out) + .5).int()
                    num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
    
                if num_total == 0:
                    num_total = 1
    
                test_running_loss /= num_total
                test_acc = 100 * (num_correct / num_total)
                LOGGER.info(f"Epoch [{epoch+1}/{self.epochs}]: test/{logging_prefix}__loss: {test_running_loss}, test/{logging_prefix}__accuracy: {test_acc}")


                if best_model is None or test_acc > best_acc:
                    best_acc = test_acc
                    best_model = deepcopy(model.state_dict())
    
                LOGGER.info(
                    f"[{epoch:04d}]: {running_loss} -- train acc: {train_accuracy} -- test_acc: {test_acc}")

        model.load_state_dict(best_model)
        return model