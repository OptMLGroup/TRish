# TRish: A Stochastic Trust Region Algorithm Based on Careful Step Normalization
Authors: Rui Shi and Frank E. Curtis

This repo is the implementation of my paper "A stochastic trust region algorithm with careful step normalization". If you are working on stochastic optimization, please cite our work.

```
@article{curtis2017stochastic,
  title={A Stochastic Trust Region Algorithm},
  author={Curtis, Frank E and Scheinberg, Katya and Shi, Rui},
  journal={arXiv preprint arXiv:1712.10277},
  year={2017}
}
```



## Introduction
This is a Python code for solving a classification problem on MNIST dataset using convolutional neural networks(CNN). The detailed structure of the network is in trishMNIST.py  The python code will compare three algorithm: Stochastic Gradient Descent (algorithm == 0), TRish(algorithm == 1) and Adagrad(algorithm == 2). The tuning scheme is described in my paper "A stochastic trust region algorithm with careful step normalization" for SG and TRish. Adagrad follows similar tuning scheme for SG.

By default, the code will tune 60 sets of hyperparameters and pick the one with highest training accuracy. After that it will have 10 test runs on testing set. You can also modify the parameters in the code after reading the parameter description.




## Usage Guide
First, you need to enter TRish_MNIST directory by  ``` cd TRish_MNIST```
The algorithms can be run using the syntax: ``` python runMNIST.py```

### Dependencies
* Numpy
* [TensorFlow](https://www.tensorflow.org/)>=1.2

### Parameters
The parameters for the **runMNIST** are:
- ```num_epochs```: number of epochs need to run (default ```num_epochs=10```)
- ```save_last```: number of last result need to save(default ```save_last=10```)
- ```values_to_save```: number of points need to save in one run(default ```values_to_save=100```)
- ```experiments```: number of runs(default ```experiments=5```)
- ```batch_size```: batch_size to obtain the stochastic gradient(default ```batch_size=5```)
- ```(alpha_min_exp,alpha_max_exp,alpha_exps)```: Tuning for SG stepsizes from 10^alpha_min_exp to 10^alpha_max_exp with exponential base alpha_exps
- ```(gamma1_choices,gamma2_choices)```: Tuning set for TRish is set as gamma1 = gamma1_choices/G and gamma2 = gamma2_choices/G


The extra parameters for the **trishMNIST** are:
- ```algorithm```: =0 SG; =1 TRish; =2 Adagrad
- ```save_data```: =1 Save output to file; =0 No save output to file
- ```num_runs```:  Number of total runs the algorithm takes

The output for the **trishMNIST** are:
- ```output```: The output of the algorithm
- ```best_output```: The best output of the algorithm
- ```parameters```: The parameters for the algorithm
- ```G```: Estimate for the norm of the gradient
The random seed is default to start at 6. To modify random seed, you will need to modify line 186: numpy.random.seed(run_number+5). 
run_number is default to start at 1 and end at num_runs.








