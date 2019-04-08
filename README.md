# TRish: A Stochastic Trust Region Algorithm Based on Careful Step Normalization
Authors: Rui Shi and Frank E. Curtis

This repo is the implementation of my paper "A stochastic trust region algorithm with careful step normalization". To cite the paper

## Introduction
This is a Python code for solving a classification problem on MNIST dataset using convolutional neural networks(CNN). The detailed structure of the network is in trishMNIST.py  The python code will compare three algorithm: Stochastic Gradient Descent (algorithm == 0), TRish(algorithm == 1) and Adagrad(algorithm == 2). The tuning scheme is described in my paper "A stochastic trust region algorithm with careful step normalization" for SG and TRish. Adagrad follows similar tuning scheme for SG.

By default, the code will tune 60 sets of hyperparameters and pick the one with highest training accuracy. After that it will have 10 test runs on testing set. You can also modify the parameters in the code after reading the parameter description.




## Usage Guide
First, you need to enter TRish_MNIST directory by  ``` cd TRish_MNIST```
The algorithms can be run using the syntax: ``` python runMNIST.py```

### Dependencies
* Numpy
* [TensorFlow](https://www.tensorflow.org/)>=1.2

The parameters for the **runMNIST** are:
