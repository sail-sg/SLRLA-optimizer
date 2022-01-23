# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, random
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import sys, pathlib
def get_dataset(name, batch_size, num_workers, input_norm, data_path='./Data/CIFAR'):
    if name == 'CIFAR10':
        # from .CIFAR import CIFAR10
        loader = CIFAR10(batch_size, num_workers, input_norm, None, data_path)
        train_loader = loader.train
        test_loader = loader.test
    elif name == 'CIFAR100':
        # from .CIFAR import CIFAR100
        loader = CIFAR100(batch_size, num_workers, input_norm, data_path)
        train_loader = loader.train
        test_loader = loader.test
    elif name == 'ImageNet':
        # from .ImageNet import ImageNet
        loader = ImageNet(batch_size, num_workers, input_norm, data_path)
        train_loader = loader.train
        test_loader = loader.test
    else:
        assert False, 'Invalid Datasets !'
    return loader, train_loader, test_loader


ROOT = os.path.join(pathlib.Path.home(), 'Data/benchmark')
DATA_PATH = os.path.join(ROOT, 'CIFAR')

class CIFAR10():
    def __init__(self, batch_size, threads, is_norm, sampling_classes=None, data_path='./Data/CIFAR'):
        self.img_channels = 3
        if is_norm:
            self.mean = [0.4914, 0.4822, 0.4465]; self.std = [0.2023, 0.1994, 0.2010]
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                # Cutout()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Cutout()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
        if sampling_classes is not None:
            train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
            print('Classification between .....', sampling_classes)
            indices_train = self.get_indices(train_set, sampling_classes)
            indices_test = self.get_indices(test_set, sampling_classes)
            self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads, sampler = torch.utils.data.SubsetRandomSampler(indices_train))
            self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, num_workers=threads, sampler = torch.utils.data.SubsetRandomSampler(indices_test))
            self.num_classes = len(sampling_classes)
        else:
            train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
            self.num_classes = 10
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
            self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=threads)

    def get_indices(self, dataset, class_name):
        assert isinstance(class_name, list)
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] in class_name:
                indices.append(i)
        return indices

    
class CIFAR100():
    def __init__(self, batch_size, threads, is_norm, data_path='./Data/CIFAR'):
        self.img_channels = 3
        if is_norm:
            self.mean=[0.5071, 0.4867, 0.4408]; self.std=[0.2675, 0.2565, 0.2761]
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                # Cutout()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Cutout()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=test_transform)
        
        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.num_classes = 100




class ImageNet():
    def __init__(self, batch_size=256, threads=8, is_norm=None, 
                 data_path='/Public/Dataset/imagenetData/ILSVRC/Data/CLS-LOC'):
        self.img_channels = 3
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')
    
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]))
        self.train = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=threads, pin_memory=True)
        
        val_dataset =  datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]))
        self.test = torch.utils.data.DataLoader(val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=threads, pin_memory=True)
        
        self.num_classes = 1000
