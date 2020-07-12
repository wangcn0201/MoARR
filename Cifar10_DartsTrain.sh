#!/bin/sh


python Cifar10_DartsTrain.py \
--code "[[0, 2, 0, 1, 2], [3, 1, 0, 1, 8], [7, 0, 0, 0, 4], [2, 0, 1, 1, 0]]" \
--dataname CIFAR-10 

python Cifar10_DartsTrain.py \
--code "[[1, 2, 0, 2, 4], [4, 1, 0, 2, 8], [4, 0, 0, 0, 5], [4, 0, 0, 1, 0]]" \
--dataname CIFAR-10 

