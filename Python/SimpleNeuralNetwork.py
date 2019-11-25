'''
In this file we provide a simple implementation of a neural network. 
We use a sigmoid as activation function and lean square as loss function. 

This impleentation is not efficient for a large number of neurons and should 
thus not be used for real-world applications. Much more efficient 
implementations are provided, for instance, by the Tensorflow and Keras 
libraries. The main goal of the present file is to show the inner workings of a
basic neural network in a more transparent way.
'''

import numpy as np


def sigmoid(x): 
	return 1. / (1. + np.exp(-x))


class NeuralNetwork: 
	'''
	Assume the last layer has exactly one neuron
	Assume the first hidden layer as has many neurons as there are inputs
	We use that the derivative of the sigmoid is f' = (1-f) f
	'''
	def __init__(self, layers, weights = None, bias = None, learn_rate = 0.1, epochs = 1000):
		
		self.layers = layers
		self.learn_rate = learn_rate
		self.epochs = epochs
		
		# If the bias are not given, choose them randomly
		if bias is None:
			bias = []
			for i in range(len(layers)):
				bias.append([])
				for j in range(layers[i]):
					bias[-1].append(np.random.normal())
		
		# If the weights are not given, choose them randomly
		if weights is None:
			weights = []
			for i in range(len(layers)):
				weights.append([])
				if i > 0:
					for j in range(layers[i]):
						weights[-1].append([np.random.normal() for k in range(layers[i-1])])
				else: 
					for j in range(layers[i]):
						weights[-1].append([np.random.normal() for k in range(layers[0])])

		self.weights = weights
		self.bias = bias

	def feedforward(self, x):
		for i in range(len(layers)):
			x = [sigmoid(np.dot(self.weights[i][j], x) + self.bias[i][j]) for j in range(self.layers[i])]
		return x

	def train(self, data, y_true_all, learn_rate = 0, epochs = 0):

		# If the learning rate and number of epochs are not given explicitly, take those of the instance
		if learn_rate == 0:
			learn_rate = self.learn_rate
		if epochs == 0:
			epochs = self.epochs

		# to avoid writing "self" all the time
		layers = self.layers
		weights = self.weights
		bias = self.bias

		for epoch in range(epochs):
			for x, y_true in zip(data, y_true_all): # run over the data

				# feedfoward, retaining the state of each neuron
				states = [x[:]]
				for i in range(len(layers)):
					x = [sigmoid(np.dot(weights[i][j], x) + bias[i][j]) for j in range(layers[i])]
					states.append(x[:])
				
				# the predicted value is the state of the last neuron
				y_pred = states[-1][0]

				# derivative of the loss function with respect to y_pred
				d_L_d_ypred = 2 * (y_pred - y_true)
				
				# partial derivatives for the output layer
				state = states[-1][0]
				# partial derivative with respect to the input
				d_ypred_d_x = [[[weights[-1][0][k]*state*(1.-state) for k in range(layers[-2])]]]
				# partial derivative with respect to the weights
				d_ypred_d_weights = [[[states[-2][k]*state*(1.-state) for k in range(layers[-2])]]]
				# partial derivative with respect to the bias
				d_ypred_d_bias = [[state*(1.-state)]]

				# partial derivatives for the other layers
				for i in range(2, len(layers)+1): # running backward over the layers
					d_ypred_d_x.insert(0,[])
					d_ypred_d_weights.insert(0,[])
					d_ypred_d_bias.insert(0,[])
					for j in range(layers[-i]):
						# derivative of y_pred with  respect to the output of this neuron
						d_ypred_d_yint = np.sum([d_ypred_d_x[1][k][j] for k in range(layers[-i+1])])
						state = states[-i][j]
						d_ypred_d_x[0].append([weights[-i][j][k]*state*(1.-state)*d_ypred_d_yint 
							for k in range(len(weights[-i][j]))])
						d_ypred_d_weights[0].append([states[-i-1][k]*state*(1.-state)*d_ypred_d_yint 
							for k in range(len(weights[-i][j]))])
						d_ypred_d_bias[0].append(state*(1.-state)*d_ypred_d_yint)

				# update weights and bias
				for i in range(len(layers)):
					for j in range(layers[i]):
						for k in range(len(weights[i][j])): 
							weights[i][j][k] -= learn_rate * d_L_d_ypred * d_ypred_d_weights[i][j][k]
						bias[i][j] -= learn_rate * d_L_d_ypred * d_ypred_d_bias[i][j]

	# evaluate the loss function
	def loss(self, data, y_true_all):
		res = 0.
		for x, y_true in zip(data, y_true_all):
			res = res + (self.feedforward(x) - y_true)**2
		return res[0]/len(data)
	
	# save the neural networ in a npy file
	def save(self, name_file):
		np.save(name_file+'.npy', [self.layers, self.weights, self.bias, 
		         self.learn_rate, self.epochs])

# load a neural network from a npy file
def load_NN_1(name_file):
	list_parameters = np.load(name_file+'.npy', allow_pickle=True)
	return NeuralNetwork_1(*list_parameters)


# Example use (data taken from https://victorzhou.com/blog/intro-to-neural-networks/): 

layers = [2, 1]
data = np.array([
  [-2, -1],  
  [25, 6],   
  [17, 4],   
  [-15, -6], 
])
y_true_all = np.array([
  1, 
  0, 
  0, 
  1, 
])

myNetwork = NeuralNetwork(layers)
myNetwork.train(data, y_true_all)
print(myNetwork.loss(data, y_true_all))
