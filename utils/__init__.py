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

import random
import numpy as np
import torch
import logging
import os
import sys
import time
import pathlib
import math
import os

print(os.path.dirname(os.path.realpath(__file__)))

def get_input_output_dim(dataset):
    if dataset == 'CIFAR10' or dataset == 'SVHN':
        return 3, 10
    elif dataset == 'FMNIST' or dataset == 'MNIST':
        return 1, 10


def makedirs(dirname):
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)


def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info('\n\n------ ******* ------ New Log ------ ******* ------')
    return logger


class get_epoch_logger():
    def __init__(self):
        self.epochs = []
        self.results = []
        self.best_epoch = 0; self.best_result = 0
    def append_results(self, list):
        self.epochs.append(list[0])
        self.results.append(list[1])
    def update_best_epoch(self):
        if self.results[-1] >= self.best_result:
            self.best_epoch = self.epochs[-1]
            self.best_result = self.results[-1]
        message = 'Best result @ {:03d}, {}.'.format(self.best_epoch, self.best_result)
        return self.best_epoch, message
    def update_best_epoch_to_logger(self, logger):
        if self.results[-1] >= self.best_result:
            self.best_epoch = self.epochs[-1]
            self.best_result = self.results[-1]
        logger.info('Best result @ {:03d}, {}.'.format(self.best_epoch, self.best_result))
        return self.best_epoch


class timer():
    def __init__(self):
        self.tic()
        
    def tic(self):
        self.t0 = time.time()
        # print(time.strftime('%Y-%m-%d-%H:%M:%S'))
        
    def toc(self, restart=True):
        diff = time.time()-self.t0
        if restart: self.t0 = time.time()
        return diff

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as figure
plt.rcParams["font.family"] = "serif"


def subplot_figure(root, file, data_plots):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
            
            e.g.: data = [subplot_1, subplit_2,]
    """
    assert isinstance(data_plots, list)
    num_plots = len(data_plots)
    w, h = figure.figaspect(1/2 * num_plots)
    fig = plt.figure(figsize=(w, h))
    for i in range(num_plots):
        ax = fig.add_subplot(num_plots, 1, i+1, frameon=True)
        axplot_multi_lines(ax=ax, data=data_plots[i])    
    fig.tight_layout()
    fig.savefig(os.path.join(root,file), dpi=200)
    plt.close()

    
def axplot_multi_lines(ax, data):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
            
            e.g.: data = [
                        {title':'', 'x_label':'', 'y_label':'', 'y_axis':[0,1],},
                        {'x':, 'y':, 'color':'blue', 'label':'nat.', 'linewidth':0.5},
                        {'x':, 'y':, 'color':'tomato', 'label':'adv.', 'linewidth':0.5}
                    ]
    """
    assert isinstance(data, list)
    for i in range(1,len(data)):
        line = data[i]
        x = line['x']
        y = line['y']
        color = line['color']
        label = line['label']
        linewidth=line['linewidth']
        ax.plot(x, y, '-', color=color, alpha=0.8/math.sqrt(i+1), linewidth=linewidth, label=label)
        # ax.fill_between(x, y, np.zeros_like(x), facecolor=color, alpha=0.8/math.sqrt(i+1), label=label)
    ax.legend(loc='upper center', markerscale=1, fancybox=True, fontsize=12)

    title = data[0]['title']
    x_label = data[0]['x_label']
    y_label = data[0]['y_label']
    y_axis = data[0]['y_axis']
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_axis is not None:
        ax.set_ylim(min(y_axis[0]), max(y_axis[1]))

    ax.margins(x=0.01)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set(facecolor = "ivory")
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=.5)
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=.5)
    
    

if __name__ == "__main__":
    pass
