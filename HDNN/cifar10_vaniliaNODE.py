"""
Train a Vanilia NODE on CIFAR10 dataset.
python3 cifar10_vaniliaNODE.py --net_layer_num 16 --conv_channel_num 128 --h 0.1  --lr 1e-3 --weight_decay 1e-3 --max_epochs 3 --scheduler_milestones=[1, 2, 3] --scheduler_gamma=0.1
"""

"""
this script implements the CIFAR10 classification task using the NODE+Euler discretization method
by using  the following tricks, we can show that the contraction regularization improves the robustness to Guasssian noise
- proper weight initializaton
- smooth leaky relu with slop 0.1
- porper optimization parameters
"""




from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import matplotlib.pyplot as plt
from tomlkit import comment
from pytorch_lightning import LightningModule
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.nn import functional as F
import torchvision.transforms as T
import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
class Net(nn.Module):
    def __init__(self, net_layer_num, conv_channel_num, h):
        super().__init__()
        self.net_layer_num = net_layer_num
        self.conv_channel_num = conv_channel_num

        # output of self.conv0 self.conv_channel_num*32*32
        self.conv0 = nn.Conv2d(3, self.conv_channel_num, 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.conv_channel_num)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.conv = nn.ModuleList([nn.Conv2d(self.conv_channel_num,
                                             self.conv_channel_num, 3, 1, 1) for i in range(self.net_layer_num)])  # output of self.conv[i] is self.conv_channel_num*32*32

        self.fc1 = nn.Linear(self.conv_channel_num*32*32, 10)
        self.h = h

    def smooth_leaky_relu(self, x):
        alpha = 0.1
        return alpha*x+(1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn(x)
        x = self.dropout1(x)
        for i in range(self.net_layer_num):
            x = x+self.h*self.smooth_leaky_relu(self.conv[i](x))

        x = torch.flatten(x, 1)
        x = self.dropout2(x)
        logits = self.fc1(x)
        return logits


class CIFAR10Classifier(LightningModule):
    def __init__(self, net_layer_num, conv_channel_num, h, lr, weight_decay, scheduler_milestones, scheduler_gamma):
        super().__init__()
        self.net_layer_num = net_layer_num
        self.conv_channel_num = conv_channel_num
        self.h = h
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_milestones = scheduler_milestones
        self.scheduler_gamma = scheduler_gamma
        self.neural_net = Net(net_layer_num, conv_channel_num, h)
        self.acc = Accuracy()

    def forward(self, x):
        return self.neural_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        train_loss = F.cross_entropy(logits, y.long())
        self.log("train_loss", train_loss)
        self.log("optimizer_lr", self.optimizer.param_groups[0]["lr"])
        return train_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.acc(logits, y)
        self.log("test_acc", self.acc)
        self.log("test_loss", loss)
        self.log("hp_metric", self.acc)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.neural_net.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=self.scheduler_milestones, gamma=self.scheduler_gamma)
        return [self.optimizer], [self.scheduler]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--net_layer_num', type=int, default=16)
    parser.add_argument('--conv_channel_num', type=int, default=128)
    parser.add_argument('--h', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler_milestones',
                        type=int, nargs='+', default=[50, 70, 80])
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    args = parser.parse_args()

    # define NODE
    logger = TensorBoardLogger(
        "tb_logs", name="cifar10_vaniliaNODE")
    logger.log_hyperparams({"net_layer_num": args.net_layer_num, "conv_channel_num": args.conv_channel_num,
                           "h": args.h, "max_epochs": args.max_epochs, "lr": args.lr, "weight_decay": args.weight_decay, "scheduler_milestones": args.scheduler_milestones, "scheduler_gamma": args.scheduler_gamma})

    NODE = CIFAR10Classifier(net_layer_num=args.net_layer_num, conv_channel_num=args.conv_channel_num, h=args.h,
                             lr=args.lr, weight_decay=args.weight_decay, scheduler_milestones=args.scheduler_milestones, scheduler_gamma=args.scheduler_gamma)
    logger.log_graph(NODE)

    trainer = pl.Trainer(gpus=[1, 2, 3], num_nodes=1,
                         callbacks=[], max_epochs=args.max_epochs, logger=logger, gradient_clip_val=0.5)

    # generate data set
    # the transform follows the example here: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    train_dataloader = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=True, download=True,
                                                                    transform=train_transforms),  batch_size=128, shuffle=True)

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_dataloader = torch.utils.data.DataLoader(datasets.CIFAR10(
        'data', train=False, transform=test_transforms), batch_size=128, shuffle=True)

    # train neural networks

    trainer.fit(NODE, train_dataloader)

    # test performance
    train_accuracy = trainer.test(NODE, train_dataloader)
    logger.log_metrics({"train_accuracy": train_accuracy[0]["test_acc"]})
    logger.log_metrics({"train_loss": train_accuracy[0]["test_loss"]})
    test_accuracy = trainer.test(NODE, test_dataloader)
    logger.log_metrics({"test_accuracy": test_accuracy[0]["test_acc"]})
    logger.log_metrics({"test_loss": test_accuracy[0]["test_loss"]})
