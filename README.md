# MoARR: Multi-Objective Neural Architecture Search Based on Diverse Structures and Adaptive Recommendation

Multi-Objective Neural Architecture Search Based on Diverse Structures and Adaptive Recommendation<br>
Chunnan Wang, Hongzhi Wang, Guocheng Feng, Fei Geng<br>
https://arxiv.org/abs/2007.02749<br>

# Requirements

``Python >= 3.5.5, PyTorch >= 1.1.0, torchvision >= 0.2.0, CUDA >= 10.0, cuDNN >= 7.5``

# Reproducing the results on CIFAR-10

``sh Cifar10_ModelLoad.sh``<br>
* Expected result: 2.48% test error rate with 2.1M model parameters and 0.38G flops.<br>
* Expected result: 2.62% test error rate with 1.3M model parameters and 0.33G flops.<br>

# Training the CNN code on CIFAR-10

``sh Cifar10_DartsTrain.sh``<br>
