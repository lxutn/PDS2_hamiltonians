
from datasets import CIFAR_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchmetrics.functional import accuracy
from networks import Network
import params as p
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.loggers import TensorBoardLogger


class LitResnet(LightningModule):
    def __init__(self, learn_params, net_params, n_labels, img_size, device):
        super().__init__()
        self.network = Network(net_params, n_labels, img_size, device)
        self.criterion = nn.CrossEntropyLoss()
        self.learn_params = learn_params
        self.save_hyperparameters()

    def forward(self, x):
        out = self.network(x)
        return out

    def regularization(self):
        a = 0.1
        alpha = 0.1
        b = 1
        regularizer = torch.zeros(1, requires_grad=True).to(self.device)
        for unit in self.network.units:
            # batchnorm layer do not have K or b
            if not isinstance(unit, nn.BatchNorm2d):
                for j in range(unit.node_params.n_blocks):
                    conv1weight = unit.convList[j].weight
                    Nc = conv1weight.size()[0]
                    for i in range(Nc):
                        regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv1weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv1weight[i, :, :, :]))+torch.sum(
                            torch.abs(conv1weight[:, i, :, :]))))
        return regularizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("classification_error_loss", loss)
        if self.learn_params.contraction_regularization:
            regularizer = self.regularization()
            loss = loss+regularizer
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
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learn_params.lr,
            # momentum=self.learn_params.momentum,
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
    learn_params = p.learn_params
    net_params = p.net_params
    DATASET = "CIFAR10"
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:1" if use_cuda else "cpu")
    logger = TensorBoardLogger("tb_logs", name="neural ode")
    train_loader, test_loader, noisy_test_loader, img_size, n_labels, n_train_img = CIFAR_dataset(DATASET,
                                                                                                  '../data', learn_params.batch_size, learn_params.crop_and_flip, kwargs)

    model = LitResnet(learn_params, net_params,
                      n_labels, img_size, device)

    trainer = Trainer(
        max_epochs=learn_params.training_epochs,
        accelerator="gpu",
        devices=[1, 2, 3],
        num_nodes=1,
        logger=logger,
        callbacks=[LearningRateMonitor(
            logging_interval="step"), TQDMProgressBar(refresh_rate=1)],
    )
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("example.ckpt")
    new_model = model.load_from_checkpoint(checkpoint_path="example.ckpt")
    trainer.test(model, test_loader)
    # trainer.test(model, noisy_test_loader)
