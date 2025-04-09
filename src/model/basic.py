from time import time
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import pandas as pd


class BasicModel(pl.LightningModule):
    def __init__(self, model, optimizer, loss_fn, lr_scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss)
