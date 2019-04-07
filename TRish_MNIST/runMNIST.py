           # Import packages
import numpy
from trishMNIST import trishMNIST
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
from load import LoadAndPlot

# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Set number of epochs
num_epochs = 10

# Set number of values to save
save_last      = 10
values_to_save = 100

# Set number of experiments
experiments = 5

# Set batch size
batch_size = 128

# Run SG to compute average stochastic gradient norm
(output,best_out,parameters,G) = trishMNIST(mnist, 0, num_epochs, batch_size, 1, -3, -3, 1, [1.0], [1.0], 0, save_last, values_to_save)

# Set tuning parameters
alpha_min_exp  = -3
alpha_max_exp  =  0
alpha_exps     = 4
gamma1_choices = [4,8,16]
gamma2_choices = [0.5,0.25,0.125]
# gamma1_choices = [4]
# gamma2_choices = [0.5]

# Update gamma choices
gamma1_choices = gamma1_choices/G
gamma2_choices = gamma2_choices/G


# Reset values for SGA tuning
alpha_min_exp  = -3
alpha_max_exp  =  2
alpha_exps     = 4
alpha_min_exp = alpha_min_exp + numpy.log10(min(gamma2_choices))
alpha_max_exp = alpha_max_exp + numpy.log10(max(gamma1_choices))
alpha_exps    = alpha_exps * len(gamma1_choices) * len(gamma2_choices)
print 10**alpha_min_exp,10**alpha_max_exp
numpy.array([10**alpha_min_exp, 10**alpha_max_exp,G]).tofile('./output/sg_range',sep="")

# Tune SG_adaptive
print('Running SG Adaptive for tuning:')
(output,best_out,parameters_sga,G) = trishMNIST(mnist, 2, num_epochs, batch_size, 1, alpha_min_exp, alpha_max_exp, alpha_exps, [1.0], [1.0], 0, save_last, values_to_save)
output.tofile('./output/tune_sgada',sep="")
parameters_sga.tofile('./output/tune_sga_param',sep="")


# Tune TRish
print('Running TRish for tuning:')
(output,best_out,parameters_tr,G) = trishMNIST(mnist, 1, num_epochs, batch_size, 1, alpha_min_exp, alpha_max_exp, alpha_exps, gamma1_choices, gamma2_choices, 0, save_last, values_to_save)
output.tofile('./output/tune_trish',sep="")
parameters_tr.tofile('./output/tune_trish_param',sep="")
print output
print parameters_tr



# Tune SG
alpha_min_exp  = -3
alpha_max_exp  =  0
alpha_exps     = 4
alpha_min_exp = alpha_min_exp + numpy.log10(min(gamma2_choices))
alpha_max_exp = alpha_max_exp + numpy.log10(max(gamma1_choices))
alpha_exps    = alpha_exps * len(gamma1_choices) * len(gamma2_choices)
print 10**alpha_min_exp,10**alpha_max_exp
numpy.array([10**alpha_min_exp, 10**alpha_max_exp,G]).tofile('./output/sg_range',sep="")
print('Running SG for tuning:')
(output,best_out,parameters_sg,G) = trishMNIST(mnist, 0, num_epochs, batch_size, 1, alpha_min_exp, alpha_max_exp, alpha_exps, [1.0], [1.0], 0, save_last, values_to_save)
output.tofile('./output/tune_sg',sep="")
parameters_sg.tofile('./output/tune_sg_param',sep="")
# print output
# print parameters_sg


# Run experiments
print('Running TRish experiments:')
(output_tr,best_out_tr,parameters_tr,G) = trishMNIST(mnist, 1, num_epochs, batch_size, experiments, numpy.log10(parameters_tr[0]), numpy.log10(parameters_tr[0]), 1, [parameters_tr[1]], [parameters_tr[2]], 1, -1, values_to_save)
print('Running SG experiments:')
(output_sg,best_out_sg,parameters_sg,G) = trishMNIST(mnist, 0, num_epochs, batch_size, experiments, numpy.log10(parameters_sg[0]), numpy.log10(parameters_sg[0]), 1, [parameters_sg[1]], [parameters_sg[2]], 1, -1, values_to_save)
print('Running SG experiments:')
(output_sga,best_out_sga,parameters_sga,G) = trishMNIST(mnist, 2, num_epochs, batch_size, experiments, numpy.log10(parameters_sga[0]), numpy.log10(parameters_sga[0]), 1, [parameters_sga[1]], [parameters_sga[2]], 1, -1, values_to_save)
# Print results to file
print output_tr
output_tr.tofile('./output/output_tr',sep="")
parameters_tr.tofile('./output/parameters_tr',sep="")
output_sg.tofile('./output/output_sg',sep="")
parameters_sg.tofile('./output/parameters_sg',sep="")
output_sga.tofile('./output/output_sga',sep="")
parameters_sga.tofile('./output/parameters_sga',sep="")
LoadAndPlot(num_epochs,values_to_save,experiments)
