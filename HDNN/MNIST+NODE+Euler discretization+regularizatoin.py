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

"""
this script implements the MNIST classification task using the NODE+Euler discretization method
by using  the following tricks, we can show that the contraction regularization improves the robustness to Guasssian noise
- proper weight initializaton
- smooth leaky relu with slop 0.1
- porper optimization parameters
"""


class Net(nn.Module):
    def __init__(self, reg_flag):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1)  # output is 8*28*28
        self.conv1 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv3 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv4 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv5 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv6 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv7 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv8 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv9 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28
        self.conv10 = nn.Conv2d(8, 8, 3, 1, 1)  # output is 8*28*28

        if reg_flag:
            my_init_weight = -5*torch.eye(3)
            with torch.no_grad():
                self.conv1.weight.copy_(my_init_weight)
                self.conv2.weight.copy_(my_init_weight)
                self.conv3.weight.copy_(my_init_weight)
                self.conv4.weight.copy_(my_init_weight)
                self.conv5.weight.copy_(my_init_weight)
                self.conv6.weight.copy_(my_init_weight)
                self.conv7.weight.copy_(my_init_weight)
                self.conv8.weight.copy_(my_init_weight)
                self.conv9.weight.copy_(my_init_weight)
                self.conv10.weight.copy_(my_init_weight)

        self.fc1 = nn.Linear(6272, 10)  # 6272=8*28*28
        self.h = 0.01

    def smooth_leaky_relu(self, x):
        alpha = 0.1
        return alpha*x+(1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, x):
        x = self.conv0(x)
        x = self.smooth_leaky_relu(x)
        x = x+self.h*self.smooth_leaky_relu(self.conv1(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv2(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv3(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv4(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv5(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv6(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv7(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv8(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv9(x))
        x = x+self.h*self.smooth_leaky_relu(self.conv10(x))
        x = torch.flatten(x, 1)
        logits = self.fc1(x)
        return logits


class ImageClassifier(LightningModule):
    def __init__(self, reg_flag, lr=5e-2, gamma=0.7, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = Net(reg_flag=reg_flag)
        self.test_acc = Accuracy()
        self.reg_flag = reg_flag

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(F.log_softmax(logits), y.long())

        # below regualrize the sufficient conditions
        if self.reg_flag:
            a = 0.1
            alpha = 2
            b = 1
            conv1weight = self.model.conv1.weight
            conv2weight = self.model.conv2.weight
            conv3weight = self.model.conv3.weight
            conv4weight = self.model.conv4.weight
            conv5weight = self.model.conv5.weight
            conv6weight = self.model.conv6.weight
            conv7weight = self.model.conv7.weight
            conv8weight = self.model.conv8.weight
            conv9weight = self.model.conv9.weight
            conv10weight = self.model.conv10.weight

            Nc = conv1weight.size()[0]
            regularizer = torch.zeros(1, requires_grad=True).to(self.device)
            for i in range(Nc):
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv1weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv1weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv1weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv2weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv2weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv2weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv3weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv3weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv3weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv4weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv4weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv4weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv5weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv5weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv5weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv6weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv6weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv6weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv7weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv7weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv7weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv8weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv8weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv8weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv9weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv9weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv9weight[:, i, :, :]))))
                regularizer = regularizer + F.relu(alpha+2 * (a+b)*conv10weight[i, i, 1, 1] + b*(torch.sum(torch.abs(conv10weight[i, :, :, :]))+torch.sum(
                    torch.abs(conv10weight[:, i, :, :]))))

            self.log("regularizer", regularizer)
            self.log("loss", loss)
            return loss+regularizer
        else:
            self.log("loss", loss)
            return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(F.log_softmax(logits), y.long())
        accuracy = self.test_acc(logits, y)
        self.log("test_acc", accuracy)
        self.log("test_loss", loss)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)]


# follow the schemes in the below link to add Gaussian noises to MNIST dataset
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def display_test_and_noisy_test_images():
    images, labels = next(iter(test_dataloader))
    plt.imshow(images[0].reshape(28, 28), cmap="gray")
    plt.savefig('test dataset image.pdf')
    plt.close()

    images, labels = next(iter(noisy_test_dataloader))
    plt.imshow(images[0].reshape(28, 28), cmap="gray")
    plt.savefig('noisy test dataset image.pdf')
    plt.close()


if __name__ == "__main__":

    # display_test_and_noisy_test_images()
    output = open("./MNIST_result.txt", "a")

    experiment_numbers = 10
    node_train_accuracy_list = []
    node_test_accuracy_list = []
    node_noisy_test_accuracy_list = []

    contractive_node_train_accuracy_list = []
    contractive_node_test_accuracy_list = []
    contractive_node_noisy_test_accuracy_list = []

    for i in range(experiment_numbers):
        # define NODE and CNODE
        logger = TensorBoardLogger("tb_logs", name="neural ode")

        NODE = ImageClassifier(reg_flag=False)
        contractive_NODE = ImageClassifier(reg_flag=True)

        trainer = pl.Trainer(gpus=[1], num_nodes=1,
                             callbacks=[], max_epochs=5, logger=logger)
        # generate data set
        train_dataloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
                                                                      transform=transforms.Compose([
                                                                          transforms.ToTensor()
                                                                          # transforms.Normalize(
                                                                          #    (0.1307,), (0.3081,))
                                                                      ])),  batch_size=128, shuffle=True)

        test_dataloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=10000, shuffle=True)

        noisy_test_dataloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            AddGaussianNoise(0., 0.2)
        ])), batch_size=10000, shuffle=True)

        # train neural networks

        trainer.fit(NODE, train_dataloader)
        trainer.fit(contractive_NODE, train_dataloader)

        # test performance
        train_accuracy = trainer.test(NODE, train_dataloader)
        test_accuracy = trainer.test(NODE, test_dataloader)
        noisy_test_accuracy = trainer.test(NODE, noisy_test_dataloader)

        node_train_accuracy_list.append(
            round(train_accuracy[0]["test_acc"], 2))
        node_test_accuracy_list.append(round(test_accuracy[0]["test_acc"], 2))
        node_noisy_test_accuracy_list.append(
            round(noisy_test_accuracy[0]["test_acc"], 2))

        train_accuracy = trainer.test(contractive_NODE, train_dataloader)
        test_accuracy = trainer.test(contractive_NODE, test_dataloader)
        noisy_test_accuracy = trainer.test(
            contractive_NODE, noisy_test_dataloader)

        contractive_node_train_accuracy_list.append(
            round(train_accuracy[0]["test_acc"], 2))
        contractive_node_test_accuracy_list.append(
            round(test_accuracy[0]["test_acc"], 2))
        contractive_node_noisy_test_accuracy_list.append(
            round(noisy_test_accuracy[0]["test_acc"], 2))

    output.write('NODE \n')
    output.write("train_accuracy_list: " +
                 str(node_train_accuracy_list)+'\n')
    output.write("test_accuracy_list: " +
                 str(node_test_accuracy_list)+'\n')
    output.write("noisy_test_accuracy_list: " +
                 str(node_noisy_test_accuracy_list)+'\n')
    output.write("average_train_accuracy: " +
                 str(round(sum(node_train_accuracy_list)/experiment_numbers, 2)) + '\n')
    output.write("average_test_accuracy: " +
                 str(round(sum(node_test_accuracy_list)/experiment_numbers, 2)) + '\n')
    output.write("average_noisy_test_accuracy: " +
                 str(round(sum(node_noisy_test_accuracy_list)/experiment_numbers, 2))+'\n')
    output.write('\n')

    output.write('contractive NODE \n')
    output.write("train_accuracy_list: " +
                 str(contractive_node_train_accuracy_list)+'\n')
    output.write("test_accuracy_list: " +
                 str(contractive_node_test_accuracy_list)+'\n')
    output.write("noisy_test_accuracy_list: " +
                 str(contractive_node_noisy_test_accuracy_list)+'\n')
    output.write("average_train_accuracy: " +
                 str(round(sum(contractive_node_train_accuracy_list)/experiment_numbers, 2)) + '\n')
    output.write("average_test_accuracy: " +
                 str(round(sum(contractive_node_test_accuracy_list)/experiment_numbers, 2)) + '\n')
    output.write("average_noisy_test_accuracy: " +
                 str(round(sum(contractive_node_noisy_test_accuracy_list)/experiment_numbers, 2))+'\n')
    output.write('\n\n')
    output.close()
