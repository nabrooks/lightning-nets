import os
import setuptools

from lightning_nets.models.networks import *

import numpy as np
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning.loggers as pl_loggers

#from lightning_nets.data import *
from lightning_nets.hooks import *
from lightning_nets.modules import *
from lightning_nets.hooks.plotters import *

from torchmetrics import *

AVAIL_GPUS = min(1, torch.cuda.device_count())
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256
IMG_SIZE = 32
LATENT_SIZE = 100

class MnistDataSet(Dataset):
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5)), transforms.Resize(IMG_SIZE)])
        self.mnist_train_data = MNIST(root = 'data', train = True, transform = transform, download = True)

    def __len__(self):
        return int(len(self.mnist_train_data))

    def __getitem__(self, index):
        output, input = self.mnist_train_data.__getitem__(index)
        z = torch.randn(LATENT_SIZE).numpy()
        r = np.concatenate(([input], z))
        #z[0] = float(input)
        return torch.Tensor(r).to(dtype=torch.float32), output.to(torch.float32).numpy()#0.5*(output.numpy() + 1)

class MnistDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data/mnist", batch_size:int = BATCH_SIZE, image_size: Tuple[int, int] = [IMG_SIZE, IMG_SIZE]):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize(image_size) ])

        # download
        self.mnist_train = MnistDataSet()
        self.mnist_val = MnistDataSet()
        self.mnist_test = MnistDataSet()
        self.mnist_predict = MnistDataSet()

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        return

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.onehot = torch.zeros(10, 10)
        self.onehot = self.onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

        if torch.cuda.is_available():
            self.onehot = self.onehot.cuda()
            
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        z = input[:,1:].view(-1, 100, 1, 1)
        class_labels = input[:,0].long()
        label_one_hot = self.onehot[class_labels]#F.one_hot(class_labels, num_classes=10)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label_one_hot)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)
        self.fill = torch.zeros([10, 10, IMG_SIZE, IMG_SIZE])
        for i in range(10):
            self.fill[i, i, :, :] = 1
        
        if torch.cuda.is_available():
            self.fill = self.fill.cuda()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, output):
        label = input[:,0].long()
        label = self.fill[label]

        x = F.leaky_relu(self.conv1_1(output), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x)).view(input.shape[0], 1)

        return x

# Generator factory method used to allow gan object to create generator
# lazily (only when needed). can be exchanged easily
def generator_ctor():
    G = generator()
    G.weight_init(0,0.02)
    return G

def discriminator_ctor():
    D = discriminator()
    D.weight_init(0,0.02)
    return D

csv_logger = pl_loggers.CSVLogger(save_dir=os.getcwd(), name="logs", flush_logs_every_n_steps=3)

data_module = MnistDataModule(batch_size=BATCH_SIZE, image_size = [IMG_SIZE, IMG_SIZE])

metrics = [ MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError() ]

trainer_module = VanillaGanModule(generator_ctor(), discriminator_ctor(), batch_size=BATCH_SIZE, metrics=metrics)
plotter = MnistCganImageDataPlotter(output_dir=csv_logger.log_dir)

callback_list = [ EpochInferenceCallback(dataloader=data_module.train_dataloader(), data_plotter=plotter, num_samples=7) ]
trainer = Trainer(gpus=1, callbacks=callback_list, max_epochs=30, logger=[csv_logger], gradient_clip_val=1.0, precision=32)
trainer.fit(trainer_module, data_module)
