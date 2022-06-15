
from datasets import CIFAR_dataset, STL10_dataset
import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from IPython.core.display import display
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from networks import Network
import params as p
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from pytorch_lightning.loggers import TensorBoardLogger


class LitResnet(LightningModule):
    def __init__(self, reg_flag, learn_params, net_params, n_labels, img_size, device):
        super().__init__()
        self.network = Network(net_params, n_labels, img_size, device)
        self.criterion = nn.CrossEntropyLoss()
        self.learn_params = learn_params
        self.reg_flag = reg_flag

    def forward(self, x):
        out = self.network(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("classification_error_loss", loss)
        if self.reg_flag:
            a = 0.1
            alpha = 0.1
            b = 1
            regularizer = torch.zeros(1, requires_grad=True).to(self.device)
            for unit in self.network.units:
                # batchnorm layer do not have K or b
                if not isinstance(unit, nn.BatchNorm2d):
                    K = unit.baseUnit.getK()
                    for j in range(unit.ham_params.n_blocks):
                        conv1weight = K[:, :, :, :, j]
                        Nc = K.size()[0]
                        for i in range(Nc):
                            regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv1weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv1weight[i, :, :, :]))+torch.sum(
                                torch.abs(conv1weight[:, i, :, :]))))

            loss = loss + regularizer
            self.log("regularizer", regularizer)
        else:
            loss = loss
        self.log("total_loss", loss)
        return loss

    def evaluate(self, batch, stage=True):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    # def validation_step(self, batch, batch_idx):
    #     self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.learn_params.lr,
            momentum=self.learn_params.momentum,
            weight_decay=self.learn_params.wd,
        )

        scheduler_dict = {
            "scheduler": MultiStepLR(
                optimizer,
                milestones=self.learn_params.lr_decay_at,
                gamma=self.learn_params.lr_decay,
            )
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    learn_params = p.J1ReLU_learn_params
    net_params = p.J1ReLU_net_params
    DATASET = "CIFAR10"
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:1" if use_cuda else "cpu")
    logger = TensorBoardLogger("tb_logs", name="neural ode")
    train_loader, test_loader, noisy_test_loader, img_size, n_labels, n_train_img = CIFAR_dataset(DATASET,
                                                                                                  '../data', learn_params.batch_size, learn_params.crop_and_flip, kwargs)

    # model = LitResnet(reg_flag=False, learn_params, net_params,
    #                   n_labels, img_size, device)
    # trainer = Trainer(
    #     max_epochs=5,
    #     accelerator="gpu",
    #     devices=[1, 2, 3],
    #     num_nodes=1,
    #     logger=CSVLogger(save_dir="logs/"),
    #     callbacks=[LearningRateMonitor(
    #         logging_interval="step"), TQDMProgressBar(refresh_rate=1)],
    # )
    # trainer.fit(model, train_loader)
    # trainer.test(model, test_loader)
    # trainer.test(model, noisy_test_loader)

    model = LitResnet(True, learn_params, net_params,
                      n_labels, img_size, device)

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=[1, 2, 3],
        num_nodes=1,
        logger=logger,
        callbacks=[LearningRateMonitor(
            logging_interval="step"), TQDMProgressBar(refresh_rate=1)],
    )
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    # trainer.test(model, noisy_test_loader)
