#!/usr/bin/env python
import numpy
import os
import tensorflow
from tensorflow.python.ops.gradients import gradients

# TRish function
def trishMNIST(mnist, algorithm, num_epochs, batch_size, num_runs, alpha_exp_min, alpha_exp_max, alpha_exps, gamma1_choices, gamma2_choices, save_data, save_last, values_to_save):

    GPU = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    
    # Start tensorflow session
    sess = tensorflow.InteractiveSession()

    # Define variables and network functions
    def weight_variable(shape, std=0.1):
        initial = tensorflow.truncated_normal(shape, stddev=std, seed=0)
        return tensorflow.Variable(initial)

    def bias_variable(shape, initVal=0.1):
        initial = tensorflow.constant(initVal, shape=shape)
        return tensorflow.Variable(initial)

    def conv2d(x, W):
        return tensorflow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def pool_2x2(x):
        return tensorflow.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Set training and testing sizes
    trainSize = 55000
    testSize  = 10000
    tBS       =  5000
    
    # Declare data and labels
    x  = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])
    y_ = tensorflow.placeholder(tensorflow.float32, shape=[None, 10])

    # Set network parameters
    NF  = 32
    NF2 = 64
    FC1 = 1024     

    # Set network layers
    W_conv1      = weight_variable([5, 5, 1, NF])
    b_conv1      = bias_variable([NF])
    x_image      = tensorflow.reshape(x, [-1, 28, 28, 1])
    h_conv1      = tensorflow.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1      = pool_2x2(h_conv1)
    W_conv2      = weight_variable([5, 5, NF, NF2])
    b_conv2      = bias_variable([NF2])
    h_conv2      = tensorflow.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2      = pool_2x2(h_conv2)
    W_fc1        = weight_variable([7 * 7 * NF2, FC1])
    b_fc1        = bias_variable([FC1])
    h_pool2_flat = tensorflow.reshape(h_pool2, [-1, 7 * 7 * NF2])
    h_fc1        = tensorflow.nn.relu(tensorflow.matmul(h_pool2_flat, W_fc1) + b_fc1)
    W_fc2        = weight_variable([FC1, 10])
    b_fc2        = bias_variable([10])
    y_conv       = tensorflow.matmul(h_fc1, W_fc2) + b_fc2

    # Define tensorflow functions
    cross_entropy      = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    correct_prediction = tensorflow.equal(tensorflow.argmax(y_conv, 1), tensorflow.argmax(y_, 1))
    accuracy           = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

    # Define function to compute losses
    def computeTrainingLoss():
        batches = trainSize / tBS
        loss    = 0.0
        for i in range(int(batches)):
            loss += 1/(batches+0.0)*sess.run(cross_entropy, feed_dict = { x : mnist.train.images[i*tBS:(i+1)*tBS], 
                                                                         y_ : mnist.train.labels[i*tBS:(i+1)*tBS]})
        return loss

    def computeTestingLoss():
        batches = testSize / tBS
        loss    = 0.0
        for i in range(int(batches)):
            loss += 1/(batches+0.0)*sess.run(cross_entropy, feed_dict = { x : mnist.test.images[i*tBS:(i+1)*tBS], 
                                                                         y_ : mnist.test.labels[i*tBS:(i+1)*tBS]})
        return loss

    # Define function to compute training accuracy
    def computeTrainingAccuracy():
        batches        = trainSize / tBS
        train_accuracy = 0.0
        for i in range(int(batches)):
            train_accuracy += 1/(batches+0.0)*sess.run(accuracy, feed_dict = { x : mnist.train.images[i*tBS:(i+1)*tBS], 
                                                                              y_ : mnist.train.labels[i*tBS:(i+1)*tBS]})
        return train_accuracy

    # Define function to compute testing accuracy
    def computeTestingAccuracy():
        test_accuracy = 0.0
        batches       = testSize / tBS
        for i in range(int(batches)):
            test_accuracy += 1/(batches+0.0)*sess.run(accuracy, feed_dict = { x : mnist.test.images[i*tBS:(i+1)*tBS], 
                                                                             y_ : mnist.test.labels[i*tBS:(i+1)*tBS]})
        return test_accuracy

    # Define parameters
    params = [W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2]    
    NP     = len(params)

    # Define temporary variables needed for algorithm
    s  = [ tensorflow.Variable(tensorflow.zeros(v.get_shape())) for v in params ]

    # Define starting point
    x0 = [ tensorflow.Variable(tensorflow.zeros(v.get_shape())) for v in params ]

    # Set tensorflow random seed
    tensorflow.set_random_seed(1)

    # Initialize all variables
    sess.run(tensorflow.global_variables_initializer())

    # Define initial point functions
    def SAVE_INITIAL_POINT():
        sess.run( [x0[i].assign(params[i]) for i in range(NP)] )
    def LOAD_INITIAL_POINT():
        sess.run( [params[i].assign(x0[i]) for i in range(NP)] )

    # Save initial point
    SAVE_INITIAL_POINT()

    # Define scaling and shifting placeholders
    SCALE = tensorflow.placeholder(tensorflow.float32)

    # Negative stochastic gradient (NSG) operator
    NSG = gradients(tensorflow.negative(cross_entropy),params)

    # Set NSG to s
    setSToNSG = [s[i].assign(NSG[i]) for i in range(NP)]

    # Add s to parameter
    addAlphaSToParam = [params[i].assign_add(SCALE*s[i]) for i in range(NP)]

    # Evaluation functions
    normOfS = tensorflow.sqrt(tensorflow.add_n([tensorflow.norm(s[i],ord='euclidean')**2 for i in range(NP)]))

    ###########################################################
    # Stochastic gradient and Stochastic trust region methods #
    ###########################################################

    # Set iteration limit
    iteration_limit = int(trainSize*num_epochs/batch_size)
    
    # Set sample indices
    samples = numpy.arange(trainSize)

    # Initialize return values
    output     = numpy.ndarray(shape=(num_runs,values_to_save+1,2), dtype=float, order='F')
    best_output     = numpy.ndarray(shape=(num_runs,values_to_save+1,2), dtype=float, order='F')
    output.fill(0.0)
    parameters = numpy.ndarray(shape=(7,1), dtype=float, order='F')
    parameters.fill(0.0)
    best_acc   = -1.0
    G_sum      =  0.0
    G_count    =  0
    
    # Loop over alpha exponents
    for alpha_exp in range(alpha_exps):
      
        # Set alpha
        if alpha_exps == 1:
            alpha = 10.0**alpha_exp_min
        else:
            alpha = 10.0**(alpha_exp_min + ((alpha_exp_max - alpha_exp_min)/(alpha_exps-1.0)) * alpha_exp);
        # end if
    
        # Loop over gamma1 values
        for gamma1 in gamma1_choices:
    
            # Loop over gamma2 values
            for gamma2 in gamma2_choices:

                # Print parameters (for sanity check while watching output during run)
                print('%e  %e  %e' % (alpha, gamma1, gamma2))

                # Loop over runs
                for run_number in range(num_runs):

                    # Reset random seed for repetition
                    numpy.random.seed(run_number+5)

                    # Load initial point from snapshot
                    LOAD_INITIAL_POINT()

                    # Initialize save fraction
                    save_frac = 0.0
                    counter   = -1
                    normSum   = 0.0001

                    # Run algorithm
                    for iter in numpy.arange(iteration_limit):
    
                        # Compute values for plots
                        if (save_data == 1 and iter >= save_frac*iteration_limit) or iteration_limit - iter <= save_last:
        
                            # Update save fraction
                            save_frac = save_frac + 1.0/values_to_save
                            counter   = counter + 1
                             
                            # Compute training and testing
                            output[run_number,counter,0] = computeTrainingLoss()
                            output[run_number,counter,1] = computeTestingAccuracy()
        
                        # end if

   
      
                        # Shuffle indices
                        numpy.random.shuffle(samples)
        
                        # Choose random indices
                        sub_samples = samples[0:batch_size]

                        # Set s as negative stochastic gradient
                        sess.run([tmp.op for tmp in setSToNSG], feed_dict = { x : mnist.train.images[sub_samples],
                                                                             y_ : mnist.train.labels[sub_samples]})

                        # Store norm of s
                        s_norm = sess.run(normOfS)
                        normSum += s_norm**2
        
                        # Add norm of s to output
                        G_sum   = G_sum + s_norm
                        G_count = G_count + 1

                        # Set step size
                        if algorithm == 0:
                            step_size = alpha
                        elif algorithm == 1:
                            if s_norm < 1/gamma1:
                                step_size = gamma1*alpha
                                parameters[4] = parameters[4] + 1;
                            elif 1/gamma1 <= s_norm and s_norm <= 1/gamma2:
                                step_size = alpha/s_norm
                                parameters[5] = parameters[5] + 1;
                            else:
                                step_size = gamma2*alpha
                                parameters[6] = parameters[6] + 1;
                        else:
                        	step_size = alpha/numpy.sqrt(normSum)
                            # end if
                        # end if

                        # Take stochastic gradient step
                        sess.run(addAlphaSToParam,feed_dict={SCALE : step_size})

                    # end for loop over iterations
                    
                    # Update best parameters
                    if (not numpy.isnan(numpy.sum((output[0,:,1])))) and numpy.sum(output[0,:,1]) > best_acc:
                        best_acc      = numpy.sum(output[0,:,1])
                        best_output   = output.copy()
                        parameters[0] = alpha
                        parameters[1] = gamma1
                        parameters[2] = gamma2
                        parameters[3] = numpy.sum(output[0,:,1])
                    # end if

                # end loop over runs
                print numpy.sum(output[0,:,1]),alpha,best_acc

            # end loop over gamma2
    
        # end loop over gamma1
    
    # end loop over alpha exponents

    # Close session
    sess.close()
    
    # Evaluate average stochastic gradient norm
    G = G_sum/G_count
    print best_output
    # Return output
    return (output,best_output,parameters,G)

# end trish
