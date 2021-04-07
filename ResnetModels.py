import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ResNet18, self).__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        # kernelCount = self.resnet18.fc.out_features
        kernelCount = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(64, 32),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, classCount))

        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet18(x)
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ResNet50, self).__init__()

        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        # kernelCount = self.resnet50.fc.out_features
        kernelCount = self.resnet50.fc.in_features
        # self.resnet50.fc = nn.Linear(kernelCount, classCount)
        # self.classifier = nn.Sigmoid()

        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(64, 32),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(32, classCount))

        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet50(x)
        x = self.classifier(x)
        return x