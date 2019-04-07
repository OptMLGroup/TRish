num_epochs = 2
num_experiments = 5
num_values = 100
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
output_tr = np.fromfile('./output/output_tr',sep="").reshape(num_experiments,num_values+1,2)
output_sg = np.fromfile('./output/output_sg',sep="").reshape(num_experiments,num_values+1,2)
output_sga = np.fromfile('./output/output_sga',sep="").reshape(num_experiments,num_values+1,2)
parameters_tr = np.fromfile('./output/parameters_tr',sep="")
parameters_sg = np.fromfile('./output/parameters_sg',sep="")
parameters_sga = np.fromfile('./output/parameters_sga',sep="")
param_tune = np.fromfile('./output/sg_range',sep="")
print 'This is the parameters or Tuning'
print param_tune
print 'This is the parameters or TRish'
print parameters_tr
print 'This is the parameters or SG'
print parameters_sg
# print 'output'
# print output_tr
avg_output_tr = output_tr.mean(axis = 0)
avg_output_sg = output_sg.mean(axis = 0)

sg_loss = np.trim_zeros(avg_output_sg[:,0])
sg_acc   = np.trim_zeros(avg_output_sg[:,1])
x = np.arange(len(sg_acc))/(1.0*len(sg_acc)) * num_epochs
print x
# print 'average_output'
# print avg_output_tr
str_loss = np.trim_zeros(avg_output_tr[:,0])
str_acc = np.trim_zeros(avg_output_tr[:,1])

print  x[10:-1].shape
string_acc = "./figures/Mnist_testing_acc_{}epoch_3alg.png".format(num_epochs)
string_acc_bw = "./figures/Mnist_testing_acc_{}epoch_gray.png".format(num_epochs)
string_loss = "./figures/Mnist_training_loss_{}epoch_3alg.png".format(num_epochs)
string_loss_bw = "./figures/Mnist_training_loss_{}epoch_gray.png".format(num_epochs)


plt.figure(0)
# plt.plot(x[10:-1], str_acc[10:-1],'b-',x[10:-1], sg_acc[10:-1],'r-')
for i in range(num_experiments):
	plt.plot(x[10:100], output_tr[i,10:-1,0],'b-')
	plt.plot(x[10:100], output_sg[i,10:-1,0],'r-')
	plt.plot(x[10:100], output_sga[i,10:-1,0],'k-')
plt.legend(['TRish', 'SG', 'SGA'], loc='upper right')
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.xlim(0,float(num_epochs))
plt.savefig(string_loss)
plt.show()


# plt.figure(1)
# # plt.plot(x[10:-1], str_acc[10:-1],'b-',x[10:-1], sg_acc[10:-1],'r-')
# for i in range(num_experiments):
# 	plt.plot(x[10:100], output_tr[i,10:-1,0],'k-')
# 	plt.plot(x[10:100], output_sg[i,10:-1,0],color='gray',linestyle ='--')


# # plt.plot(x[10:100], output_tr[0,10:-1,0],'b-',x[10:100], output_tr[1,10:-1,0],'b-',
# # 	x[10:100], output_tr[2,10:-1,0],'b-')

# plt.legend(['TRish', 'SG'], loc='upper right')
# plt.ylabel('Training Loss')
# plt.xlabel('Epochs')
# plt.xlim(0,float(num_epochs))
# plt.savefig(string_loss_bw)
# plt.show()


plt.figure(2)
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
plt.savefig(string_acc)
plt.show()

# plt.figure(3)
# # plt.plot(x[10:-1], str_acc[10:-1],'b-',x[10:-1], sg_acc[10:-1],'r-')
# for i in range(num_experiments):
# 	plt.plot(x[10:100], output_tr[i,10:-1,1],'k-')
# 	plt.plot(x[10:100], output_sg[i,10:-1,1],color='gray',linestyle ='--')


# # plt.plot(x[10:100], output_tr[0,10:-1,0],'b-',x[10:100], output_tr[1,10:-1,0],'b-',
# # 	x[10:100], output_tr[2,10:-1,0],'b-')

# plt.legend(['TRish', 'SG'], loc='lower right')
# plt.ylabel('Testing accuracy')
# plt.xlabel('Epochs')
# plt.xlim(0,float(num_epochs))
# plt.savefig(string_acc_bw)
# plt.show()
# plt.figure(1)
# plt.plot(x[10:-1],str_loss[10:-1],'b-',x[10:-1], sg_loss[10:-1],'r-')
# #plt.plot(x,str_acc_mask ,'r-',x, sg_acc_mask,'b-')
# #plt.plot(x,str_acc ,'r-',x, sg_acc,'b-')
# #plt.legend(['SG', 'TRish'], loc='upper right')
# plt.legend(['TRish', 'SG'], loc='upper right')
# plt.ylabel('Training loss')
# plt.xlabel('Epochs')
# plt.xlim(0,float(num_epochs))
# plt.savefig(string_loss)
# plt.show()

# print 'output'
# print output_tr
# avg_output_tr = output_tr.mean(axis = 0)
# avg_output_sg = output_sg.mean(axis = 0)

# sg_loss = np.trim_zeros(avg_output_sg[:,0])
# sg_acc   = np.trim_zeros(avg_output_sg[:,1])
# x = np.arange(len(sg_acc))/(1.0*len(sg_acc)) * num_epochs
# # print 'average_output'
# # print avg_output_tr
# str_loss = np.trim_zeros(avg_output_tr[:,0])
# str_acc = np.trim_zeros(avg_output_tr[:,1])

# plt.figure(0)
# plt.plot(x, sg_acc,'r--',x, str_acc,'b-')
# plt.legend(['SG', 'TRish'], loc='upper right')
# plt.ylabel('Average Testing Accuracy')
# plt.xlabel('Number of Epochs')
# plt.savefig("./figures/Mnist_testing_acc.png" )
# plt.show()

# plt.figure(1)
# plt.plot(x, sg_loss,'r--',x, str_loss,'b-')
# plt.legend(['SG', 'TRish'], loc='upper right')
# plt.ylabel('Average Training Loss')
# plt.xlabel('Number of Epochs')
# plt.savefig("./figures/Mnist_training_loss.png" )
# plt.show()


# print avg_output_tr[avg_output_tr!=0]

# output_tune_tr = np.fromfile('./output/tune_trish',sep="").reshape(2,16,2)
# parameters_tune = np.fromfile('./output/tune_trish_param',sep="")
# output_tune_sg = np.fromfile('./output/tune_sg',sep="").reshape(2,16,2)
# parameters_tune_sg = np.fromfile('./output/tune_sg_param',sep="")
# print output_tune_tr

# print parameters_tune

# print output_tune_sg

# print parameters_tune_sg