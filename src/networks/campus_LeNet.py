import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CAMPUS_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 100
        self.pool = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(4, 2, 5, bias=False, padding=2)
        self.bn3= nn.BatchNorm2d(2, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(2 * 10 * 10, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CAMPUS_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 100
        self.pool = nn.MaxPool2d(4, 4)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(4, 2, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(2 * 10 * 10, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1, 2, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(8, 3, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 100), 10, 10)
        x = F.interpolate(F.leaky_relu(x), scale_factor=1)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=4)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=4)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn5(x)), scale_factor=4)
        x = self.deconv4(x)
        x = torch.sigmoid(x)

        return x
