# Stagewise Locally-Regularized LookAhead (SLRLA)
<center><b><a href="https://panzhous.github.io">Towards Understanding Why Lookahead Generalizes Better Than SGD and Beyond</a> </b><br>
Pan Zhou <sup>*</sup>, Hanshu Yan <sup>*</sup>, Xiaotong Yuan <sup>^</sup>, Jiashi Feng <sup>*</sup>, Shuicheng Yan <sup>*</sup> <br>
<sup>*</sup> Sea AI Lab, Sea Group, <sup>^</sup> Nanjing University of Information Science & Technology <br>
Neural Information Processing Systems (NeurIPS), 2021 
</center>

---

## Requirements
This repository provides codes for SLRLA.  
- Python = 3.6  
- Pytorch = 1.6  
- CUDA = 10.1  

---

## Experimental Results

Here we list experimental results on CIFAR10 and CIFAR100. Please refer to the main paper for detailed experimental settings and more results on ImageNet.

### CIFAR10
|    Optimizer            | ResNet-18     | VGG-16  | WRN-16-10 |
| ------------- | ------- | ------- | ------- |
| stagewise SGD | 95.23   | 92.13 | 95.51 |
| stagewise LA  | 95.27   | 92.38 | 95.73 |
| SLRLA         | 95.47   | 92.63 | 96.08 |

<!-- <br> -->
### CIFAR100
| Optimizer | ResNet-18     | VGG-16   | WRN-16-10 |
| ------------- | -------- | ------- | ------- |
| stagewise SGD | 78.24    | 69.97 | 78.95 |
| stagewise LA  | 78.34    | 70.2 | 79.54 |
| SLRLA         | 78.58    | 70.63 | 79.85 |


### ImageNet
| Optimizer | ResNet-18     |
| ------------- | -------- |
| stagewise SGD | 70.23    |
| stagewise LA  | 70.30   |
| SLRLA         | 70.47    |

### Run scripts
<!-- <code>  -->
<pre>
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset CIFAR10 --network ResNet-18 --input_norm --exp_name sgd_vanilla --wd 1e-3 --lr 1e-1

CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --network ResNet-18 --input_norm --exp_name sgd_LA --wd 1e-3 --lr 1e-1 --lookahead 5_0.8

CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --network ResNet-18 --input_norm --exp_name sgd_SLRLA --wd 1e-3 --lr 1e-1 --lookahead 5_0.8 --slr 5_0.2

</pre>
<!-- </code> -->

---
## LICENSE
This repo is under the Apache-2.0 license. For commercial use, please contact the authors.

## Bibtex
> @inproceedings{Zhou2021LA,  
author = {Pan Zhou and Hanshu Yan and Xiaotong Yuan and 
Jiashi Feng and Shuicheng Yan}  
title = {Towards Understanding Why Lookahead Generalizes 
Better Than SGD and Beyond},  
booktitle = {NeurIPS},  
year = {2021}  
}
