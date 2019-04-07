def LoadAndPlot(num_epochs,num_values,num_experiments):

	import numpy as np
	import matplotlib as plt
	import matplotlib
	matplotlib.use('Agg')
#  matplotlib.use('Agg')plt.switch_backend('Agg')
	# plt.switch_backend('Agg')
	import matplotlib.pyplot as plt
	output_tr = np.fromfile('./output/output_tr',sep="").reshape(num_experiments,num_values+1,2)
	output_sg = np.fromfile('./output/output_sg',sep="").reshape(num_experiments,num_values+1,2)
	output_sga = np.fromfile('./output/output_sga',sep="").reshape(num_experiments,num_values+1,2)
	parameters_tr = np.fromfile('./output/parameters_tr',sep="")
	parameters_sg = np.fromfile('./output/parameters_sg',sep="")
	parameters_sga = np.fromfile('./output/parameters_sga',sep="")
	print 'This is the parameters or TRish'
	print parameters_tr
	print 'This is the parameters or SG'
	print parameters_sg
	print 'This is the parameters or SG Adaptive'
	print parameters_sga	
	# print 'output'
	# print output_tr
	avg_output_tr = output_tr.mean(axis = 0)
	avg_output_sg = output_sg.mean(axis = 0)
	avg_output_sga = output_sga.mean(axis = 0)

	sg_loss = np.trim_zeros(avg_output_sg[:,0])
	sg_acc   = np.trim_zeros(avg_output_sg[:,1])
	x = np.arange(len(sg_acc))/(1.0*len(sg_acc)) * num_epochs
	# print 'average_output'
	# print avg_output_tr
	str_loss = np.trim_zeros(avg_output_tr[:,0])
	str_acc = np.trim_zeros(avg_output_tr[:,1])
	   
	sga_loss = np.trim_zeros(avg_output_sga[:,0])
	sga_acc = np.trim_zeros(avg_output_sga[:,1])	    
	    
	string_acc = "./figures/Mnist_testing_acc_{}epoch_avg.png".format(num_epochs)
	string_acc_multi = "./figures/Mnist_testing_acc_{}epoch_3alg.png".format(num_epochs)
	string_loss = "./figures/Mnist_training_loss_{}epoch_avg.png".format(num_epochs)
	string_loss_multi = "./figures/Mnist_training_loss_{}epoch_3alg.png".format(num_epochs)
	plt.figure(0)
	plt.plot(x[10:-1], str_acc[10:-1],'b-',x[10:-1], sg_acc[10:-1],'r-', sga_acc[10:-1],'k-')
	plt.legend(['TRish', 'SG', 'SGA'], loc='lower right')
	plt.ylabel('Testing accuracy')
	plt.xlabel('Epochs')
	plt.xlim(0,float(num_epochs))
	plt.savefig(string_acc)
	plt.close()



	plt.figure(1)
	plt.plot(x[10:-1],str_loss[10:-1],'b-',x[10:-1], sg_loss[10:-1],'r-',sga_loss[10:-1],'k-')
	#plt.plot(x,str_acc_mask ,'r-',x, sg_acc_mask,'b-')
	#plt.plot(x,str_acc ,'r-',x, sg_acc,'b-')
	#plt.legend(['SG', 'TRish'], loc='upper right')
	plt.legend(['TRish', 'SG', 'SGA'], loc='upper right')
	plt.ylabel('Training loss')
	plt.xlabel('Epochs')
	plt.xlim(0,float(num_epochs))
	plt.savefig(string_loss)
	plt.close()

	plt.figure(2)
	# plt.plot(x[10:-1], str_acc[10:-1],'b-',x[10:-1], sg_acc[10:-1],'r-')
	for i in range(num_experiments):
		plt.plot(x[10:100], output_tr[i,10:-1,0],'b-')
		plt.plot(x[10:100], output_sg[i,10:-1,0],'r-')
		plt.plot(x[10:100], output_sga[i,10:-1,0],'k-')
	plt.legend(['TRish', 'SG', 'SGA'], loc='upper right')
	plt.ylabel('Training Loss')
	plt.xlabel('Epochs')
	plt.xlim(0,float(num_epochs))
	plt.savefig(string_loss_multi)
	plt.show()

	plt.figure(3)
	# plt.plot(x[10:-1], str_acc[10:-1],'b-',x[10:-1], sg_acc[10:-1],'r-')
	for i in range(num_experiments):
		plt.plot(x[10:100], output_tr[i,10:-1,1],'b-')
		plt.plot(x[10:100], output_sg[i,10:-1,1],'r-')
		plt.plot(x[10:100], output_sga[i,10:-1,1],'k-')

	# plt.plot(x[10:100], output_tr[0,10:-1,0],'b-',x[10:100], output_tr[1,10:-1,0],'b-',
	# 	x[10:100], output_tr[2,10:-1,0],'b-')

	plt.legend(['TRish', 'SG', 'SGA'], loc='lower right')
	plt.ylabel('Testing accuracy')
	plt.xlabel('Epochs')
	plt.xlim(0,float(num_epochs))
	plt.savefig(string_acc_multi)
	plt.show()

# print avg_output_tr[avg_output_tr!=0]

# output_tune_tr = np.fromfile('./output/tune_trish',sep="").reshape(2,16,2)
# parameters_tune = np.fromfile('./output/tune_trish_param',sep="")
# output_tune_sg = np.fromfile('./output/tune_sg',sep="").reshape(2,16,2)
# parameters_tune_sg = np.fromfile('./output/tune_sg_param',sep="")
# print output_tune_tr

# print parameters_tune

# print output_tune_sg

# print parameters_tune_sg