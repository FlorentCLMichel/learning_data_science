'''
Application of the CNN from CNN1_cpp to the MNIST database
This CNN is not optimal
It seems to be overfitted to the training data
'''

import os
import mnist
import numpy as np
import CNN1_cpp

save_file_name = 'MNIST_CNN_1.cnn'

# parameters
n_train_images = 60000 # number of images for the training
n_epochs = 10 # number of epochs
n_test_images = 10000 # number of images for the test
n_print = 4000 # accuracy printed every n_print training steps
learn_rate = 0.005 # learning rate

def normalize(image):
	return image / 255 - 0.5

# training images and labels
train_images = normalize(mnist.train_images()[:n_train_images])
train_labels = mnist.train_labels()[:n_train_images]

# test images and labels
test_images = normalize(mnist.test_images()[:n_test_images])
test_labels = mnist.test_labels()[:n_test_images]

# CNN with 2 convolution layers and 2 fully connected layers (including the Softmax one)
# The hyperparameters are not tuned—this network is thus probably not very efficient

img_w = 28 # images have size 28 by 28
img_h = 28
CL_size_filters = [5,3]
CL_num_filters = [4,8]
MP_size = [2,2]
FC_size = [20]
num_labels = 10 # 10 different possible labels

# build and train the CNN if it does not exist

if not os.path.exists(save_file_name):
	
	import matplotlib.pyplot as plt

	CNN1_1 = CNN1_cpp.CNN1(img_w, img_h, CL_size_filters, CL_num_filters, MP_size, FC_size, num_labels)

	print('MNIST CNN initialized') 

	# train the CNN

	nsteps_l = []
	loss_l = []
	acc_l = []

	print('\n\n— Start of training —')
	for epoch in range(n_epochs):
		nsteps_l.append([])
		loss_l.append([])
		acc_l.append([])
		print('\n— Epoch {:d} of {:d} —\n'.format(epoch+1, n_epochs))

		# Shuffle the training data
		permutation = np.random.permutation(len(train_images))
		train_images = train_images[permutation]
		train_labels = train_labels[permutation]

		loss = 0.
		num_correct = 0
		for i, (im, label) in enumerate(zip(train_images, train_labels)): 
			results = CNN1_1.train(train_images[i], int(train_labels[i]), learn_rate)
			loss += results[0]
			num_correct += results[1]
			if i % n_print == n_print - 1:
				nsteps_l[-1].append(i)
				loss_l[-1].append(loss / n_print)
				acc_l[-1].append(num_correct / n_print)
				print('Step {:d} — average loss: {:.10f}, accuracy: {:.3f}%'.format(i+1, loss/n_print, num_correct*100./n_print))
				loss = 0.
				num_correct = 0

	# save the CNN
	CNN1_1.save(save_file_name)

	for e in range(n_epochs):
		plt.plot(nsteps_l[e], loss_l[e], label='epoch {:d}'.format(e+1))
	plt.xlim(0, n_train_images)
	plt.xlabel(r'$n_{\mathrm{steps}}$')
	plt.ylabel(r'$\mathrm{loss}$')
	plt.grid()
	plt.legend()
	plt.tight_layout()
	plt.show()
	
	for e in range(n_epochs):
		plt.plot(nsteps_l[e], acc_l[e], label='epoch {:d}'.format(e+1))
	plt.xlim(0, n_train_images)
	plt.xlabel(r'$n_{\mathrm{steps}}$')
	plt.ylabel(r'$\mathrm{accuracy}$')
	plt.grid()
	plt.legend()
	plt.tight_layout()
	plt.show()

# load the CNN
CNN1_1 = CNN1_cpp.CNN1()
CNN1_1.load(save_file_name)

print('MNIST CNN loaded')

# test the CNN
print('\n\n— Testing the CNN —\n')
loss = 0.
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
	results = CNN1_1.forward_la(im, int(label))
	loss += results[0]
	num_correct += results[1]
print('Test Loss:', loss / n_test_images)
print('Test Accuracy:', num_correct / n_test_images)

