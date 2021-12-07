# ********************* CIFAR10

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --network ResNet-18 --input_norm \
#                             --exp_name sgd_vanilla --wd 1e-3 --lr 1e-1 \
#                             --SEED 0

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --network ResNet-18 --input_norm \
#                             --exp_name sgd_LA --wd 1e-3 --lr 1e-1 --lookahead 5_0.8 \
#                             --SEED 0

# CUDA_VISIBLE_DEVICES=2 python main.py --dataset CIFAR10 --network ResNet-18 --input_norm \
#                             --exp_name sgd_SLRLA --wd 1e-3 --lr 1e-1 --lookahead 5_0.8 --slr 5_0.2 \
#                             --SEED 0

        

