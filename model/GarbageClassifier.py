import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy
from torch.optim import Adam


class GarbageClassifier(pl.LightningModule):
    def __init__(self, num_classes,  lr=1e-3):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        self.learning_rate = lr
        self.optimizer = Adam
        self.criterion = nn.CrossEntropyLoss()
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, num_classes)
        self.accuracy = Accuracy()
        self.__dict__.update(locals())

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("tets_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)
