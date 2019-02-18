# Dependencies
import numpy as np
import os, sys
import time
import pickle

import Utilities as utils


# Define a class to hold the neural network
class multiLayerPerceptron(object):
	
	# Constructor
	def __init__(self, n_hidden = 2, hidden_dims = (1024, 2048), activation_list = ('ReLU', 'ReLU'), mode = 'train', datapath = None, model_path = None) :

		"""
		inputs :

		n_hidden : 2
			The number of hidden layers. Must be equal in size to the length of the hidden_dims
		hidden_dims : (1024, 2048)
			The tuple of output dimensions of the latent layers
		activation_list : ('relu', 'relu')
			The list of activations to be applied at the end of each of the hidden layers. The last layer is by-default 'softmax'. SUPPORT : 'ReLU', 'Sigmoid', 'Tanh', 'Softmax'
		mode : 'train'
			The mode for the neural network. SUPPORT : 'train' and 'test'
		datapath : None
			The path for loading the dataset
		model_path : None
			The path for loading the model
		"""

		# Check the hidden_dims and activations consistency
		assert(len(hidden_dims) == n_hidden)
		assert(len(activation_list) == n_hidden)

		# Create attributes
		self.hidden_dims = hidden_dims
		# Create storage for weights and biases
		self.weights_dict = {}
		self.biases_dict = {}
		self.non_linearity_dict = {}
		# Create storage for pre-activations and activations
		self.preactivation_dict = {}
		self.activation_dict = {}
		# Create storage for gradients of weights and biases
		self.grad_weights_dict = {}
		self.grad_biases_dict = {}
		# Create storage for gradients of pre-activations and activations
		self.grad_preactivation_dict = {}
		self.grad_activation_dict = {}

		# Store the input dim default
		self.input_dim = 784
		current_input_dim = self.input_dim
		# Create weight and bias matrices
		for i in range(n_hidden) :
			current_output_dim = hidden_dims[i]
			self.weights_dict['W' + str(i + 1)] = np.random.random([current_output_dim, current_input_dim]) # Randomly initialized
			self.biases_dict['b' + str(i + 1)] = np.zeros([current_output_dim, 1]) # Zero initialized
			self.non_linearity_dict['g' + str(i + 1)] = (activation_list[i], current_output_dim)
			current_input_dim = current_output_dim
		# Store the output dim default
		self.output_dim = 10
		current_output_dim = self.output_dim
		self.weights_dict['W' + str(len(self.hidden_dims) + 1)] = np.random.random([current_output_dim, current_input_dim]) # Randomly initialized
		self.biases_dict['b' + str(len(self.hidden_dims) + 1)] = np.zeros([current_output_dim, 1]) # Zero initialized
		self.non_linearity_dict['g' + str(len(self.hidden_dims) + 1)] = ('Softmax', current_output_dim)


	# Define a function to get the weights by index
	def GetWeight(self, index) :

		"""
		inputs :

		index :
			The index of the weight. Must be in {1, ..., len(self.weights_dict)}
		"""

		"""
		outputs :

		weight (implicit) :
			The weight at the index
		"""

		return self.weights_dict['W' + str(index)]


	# Define a function to set the weight by index
	def SetWeight(self, index, weight) :

		"""
		inputs :

		index :
			The index of the weight. Must be in {1, ..., len(self.weights_dict)}
		weight :
			The numpy array that needs to be stored in the location
		"""

		"""
		outputs :
		"""

		self.weights_dict['W' + str(index)] = weight


	# Define a function to get the bias by index
	def GetBias(self, index) :

		"""
		inputs :

		index :
			The index of the weight. Must be in {1, ..., len(self.weights_dict)}
		"""

		"""
		outputs :

		bias (implicit) :
			The weight at the index
		"""

		return self.biases_dict['b' + str(index)]


	# Define a function to set the bias by index
	def SetBias(self, index, bias) :

		"""
		inputs :

		index :
			The index of the weight. Must be in {1, ..., len(self.weights_dict)}
		bias :
			The numpy array that needs to be stored in the location
		"""

		"""
		outputs :
		"""

		self.biases_dict['b' + str(index)] = bias


	# Define a function to get the non-linearity by index
	def GetNonLinearity(self, index) :

		"""
		inputs :

		index :
			The index of the weight. Must be in {1, ..., len(self.weights_dict)}
		"""

		"""
		outputs :

		nonlinearity (implicit) :
			The tuple containing the name of the non-linearity and the corresponding output dimension
		"""

		return self.non_linearity_dict['g' + str(index)]


	# Define a function to initialize weights
	def initialize_weights(self, initializer, n_hidden = (1024, 2048), hidden_dims = 2) :

		"""
		inputs :

		initializer :
			The initializer for the weights. SUPPORT : 'Zero', 'Normal', 'Glorot' ('Xavier')
		n_hidden : (1024, 2048)
			The number of hidden layers
		dims : 2
			The number of dimensions
		"""

		# Check initializer type
		assert(initializer == 'Zero' or initializer == 'Normal' or initializer == 'Glorot')

		# Get the number of all the layers
		l = len(self.weights_dict)

		# Implement zero initializer
		if initializer == 'Zero' :
			for i in range(l) :
				current_shape = list(self.GetWeight(i + 1).shape)
				self.SetWeight(i + 1, np.zeros(current_shape))

		# Implement normal initializer
		elif initializer == 'Normal' :
			for i in range(l) :
				current_shape = list(self.GetWeight(i + 1).shape)
				self.SetWeight(i + 1, np.random.normal(0.0, 1.0, current_shape))

		# Implement xavier-glorot initializer
		elif initializer == 'Glorot' :
			for i in range(l) :
				current_shape = list(self.GetWeight(i + 1).shape)
				d = np.sqrt(6.0/(1e-10 + current_shape[0] + current_shape[1])) # root(6/(h_{l - 1} + h_l))
				self.SetWeight(i + 1, np.random.uniform(-1.0*d, d, current_shape))


	# Define a function to compute the forward pass of the neural network, using the current weights
	def forward(self, x) :

		"""
		inputs :

		x : 
			The input batch. Shape : [<input_dim>, <batch_size>]
		"""

		"""
		outputs :

		f_of_x :
			The output of the neural network. Shape : [<output_dim>, <batch_size>]
		"""

		# Clear the activation and pre-activation dict
		self.preactivation_dict.clear()
		self.activation_dict.clear()

		l = len(self.weights_dict) # It is going to be n_hidden + 1
		self.activation_dict['h0'] = x
		self.preactivation_dict['a0'] = x
		self.non_linearity_dict['g0'] = ('Linear', x.shape[0]) # Only for the input layer it makes sense

		# Compute the forward pass one-by-one and store all the pre-activations and activations in attribute arrays
		for i in range(l) :
			self.preactivation_dict['a' + str(i + 1)] = np.copy(  np.matmul(self.GetWeight(i + 1), self.activation_dict['h' + str(i)]) + self.GetBias(i + 1)  )
			non_lin = self.GetNonLinearity(i + 1)[0]
			if non_lin == 'Sigmoid' :
				self.activation_dict['h' + str(i + 1)] = np.copy(  utils.Sigmoid(self.preactivation_dict['a' + str(i + 1)])  )
			elif non_lin == 'Tanh' :
				self.activation_dict['h' + str(i + 1)] = np.copy(  utils.Tanh(self.preactivation_dict['a' + str(i + 1)])  )
			elif non_lin == 'Softmax' :
				self.activation_dict['h' + str(i + 1)] = np.copy(  utils.Softmax(self.preactivation_dict['a' + str(i + 1)])  )
			elif non_lin == 'ReLU' :
				self.activation_dict['h' + str(i + 1)] = np.copy(  utils.ReLU(self.preactivation_dict['a' + str(i + 1)])  )
			else :
				# The control should never reach here
				print('[ERROR] Unimplemented Non-Linearity Encountered at Layer : ', i, ' with Non-Linearity :', non_lin)
				print('[ERROR] Terminating the code ...')
				sys.exit()

		# Just-in-case
		return np.copy(self.activation_dict['h' + str(l)])


	# Define a function to compute the backward pass of the neural network, using the current weights
	def backward(self, y_true, is_retain_graph = False) :

		"""
		inputs :

		y_true :
			The ground truth labels
		is_retain_graph : False
			To keep previous gradients or not
		"""

		"""
		outputs :
		"""

		# Clear the gradients stored in the last pass, IF WE DO NOT WANT TO RETAIN
		if not is_retain_graph :
			self.grad_preactivation_dict.clear()
			self.grad_activation_dict.clear()
			self.grad_weights_dict.clear()
			self.grad_biases_dict.clear()

			# Evaluate the backward pass
			l = len(self.weights_dict) # It is going to be n_hidden + 1
			batch_size = self.activation_dict['h0'].shape[1]
			normalizer = 1.0/batch_size

			# Evaluate the last layer pre-activation gradient for softmax
			self.grad_activation_dict['grad_h' + str(l)] = None
			if self.GetNonLinearity(l)[0] == 'Softmax' :
				self.grad_preactivation_dict['grad_a' + str(l)] = self.activation_dict['h' + str(l)] - np.transpose(np.eye(self.output_dim)[y_true.astype(np.int)])
			else :
				print('[ERROR] Unimplemented Loss Criterion at Last Layer.')
				print('[ERROR] Terminating the code ...')
				sys.exit()

			# Loop over the layers
			for i in range(l, 0, -1) : # Goes till 1
				# Weight gradient 
				self.grad_weights_dict['grad_W' + str(i)] = np.copy( normalizer * np.matmul(self.grad_preactivation_dict['grad_a' + str(i)], np.transpose(self.activation_dict['h' + str(i - 1)])) ) # Using standard formula
				# Bias gradient
				self.grad_biases_dict['grad_b' + str(i)] = np.copy( np.reshape(np.mean(self.grad_preactivation_dict['grad_a' + str(i)], axis = 1), [-1, 1]) )
				# Activation gradient
				self.grad_activation_dict['grad_h' + str(i - 1)] = np.copy( np.matmul(np.transpose(self.GetWeight(i)), self.grad_preactivation_dict['grad_a' + str(i)]) )
				# Preactivation gradient 
				self.grad_preactivation_dict['grad_a' + str(i - 1)] = np.copy (self.grad_activation_dict['grad_h' + str(i - 1)] * utils.GetPointwiseGradientOfNonLinearity(self.preactivation_dict['a' + str(i - 1)], self.GetNonLinearity(i - 1)[0]) )

		# Else, only add to the current gradients
		else :

			sys.exit()

			# Evaluate the backward pass
			l = len(self.weights_dict) # It is going to be n_hidden + 1
			batch_size = self.activation_dict['h0'].shape[1]
			normalizer = 1.0/batch_size

			# Evaluate the last layer pre-activation gradient for softmax
			self.grad_activation_dict['grad_h' + str(l)] = None
			if self.GetNonLinearity(l)[0] == 'Softmax' :
				self.grad_preactivation_dict['grad_a' + str(l)] += self.activation_dict['h' + str(l)] - np.transpose(np.eye(self.output_dim)[y_true.astype(np.int)])
			else :
				print('[ERROR] Unimplemented Loss Criterion at Last Layer.')
				print('[ERROR] Terminating the code ...')
				sys.exit()

			# Loop over the layers
			for i in range(l, 0, -1) : # Goes till 1
				# Weight gradient 
				self.grad_weights_dict['grad_W' + str(i)] += normalizer * np.matmul(self.grad_preactivation_dict['grad_a' + str(i)], np.transpose(self.activation_dict['h' + str(i - 1)])) # Using standard formula
				# Bias gradient
				self.grad_biases_dict['grad_b' + str(i)] += np.reshape(np.mean(self.grad_preactivation_dict['grad_a' + str(i)], axis = 1), [-1, 1])
				# Activation gradient
				self.grad_activation_dict['grad_h' + str(i - 1)] += np.matmul(np.transpose(self.GetWeight(i)), self.grad_preactivation_dict['grad_a' + str(i)])
				# Preactivation gradient 
				self.grad_preactivation_dict['grad_a' + str(i - 1)] += self.grad_activation_dict['grad_h' + str(i - 1)] * utils.GetPointwiseGradientOfNonLinearity(self.preactivation_dict['a' + str(i - 1)], self.GetNonLinearity(i - 1)[0])


	# Define a function to update the weights based on the current gradients
	def update(self, eta) :

		"""
		inputs :

		eta :
			The step size for the updates
		"""

		"""
		outputs :
		"""

		# Update each weight
		for i in self.weights_dict :
			self.weights_dict[i] -= eta*self.grad_weights_dict['grad_' + i]
		for i in self.biases_dict :
			self.biases_dict[i] -= eta*self.grad_biases_dict['grad_' + i]


	# Define a function to compute the loss of the current forward pass
	def loss(self, y_true, y_pred = None, loss_type = 'Cross-Entropy') :

		"""
		inputs :

		y_true :
			The ground truth labels of 0.0 and 1.0. Shape : [<batch_size>]
		y_pred : None
			The predictions, if given explicitly. Otherwise, the default predictions are activations of the last layers, aka 'h' + str(len(self.weights_dict))
		loss_type : 'Cross-Entropy'
			The loss to be evaluated 
		"""

		"""
		outputs :

		loss (implicit) :
			The loss evaluated with the given criterion
		"""

		# If the y_pred is to be default, get the last layer activation (soft-max-ed)
		if y_pred is None :
			y_pred = self.activation_dict['h' + str(len(self.weights_dict))]  

		# Define the cross-entropy loss
		if loss_type == 'Cross-Entropy' :
			# Get the indices in the int format
			y_indices = np.array(y_true).astype(np.int)
			# Indices are columns and from each column, we want the corresponding indexed entry
			predictions = []
			for i in range(len(y_indices)) :
				predictions.append(y_pred[int(y_indices[i])][i])
			predictions = np.array(predictions).astype(np.float32)
			# Evaluate the loss
			return np.mean(-1.0 * np.log(1e-10 + predictions))

		else :
			print('[ERROR] Loss of the type : ', loss_type, ' is not implemented.')
			print('[ERROR] Terminating the code ...')
			sys.exit()


	# Define a function to compute the accuracy based on two inputs being equal
	def EvaluateAccuracy(self, y_true, y_pred = None) :

		"""
		inputs :

		y_true :
			The ground truth labels of 0.0 and 1.0. Shape : [<batch_size>]
		y_pred : None
			The predictions, if given explicitly. Otherwise, the default predictions are activations of the last layers, aka 'h' + str(len(self.weights_dict))
		"""

		"""
		outputs :

		accuracy : 
			The accuracy of the ground truth and the predictions
		correct_predictions :
			The 1-0 array representing the correct and incorrect predictions
		"""
		
		if y_pred is None :
			y_pred = self.activation_dict['h' + str(len(self.weights_dict))]  

		predictions = np.reshape(np.argmax(y_pred, axis = 0), [-1, 1]).astype(np.float32)
		ground_truth = np.reshape(y_true, [-1, 1]).astype(np.float32)

		correct_predictions = (predictions == ground_truth).astype(np.float32)
		accuracy = np.mean(correct_predictions)

		return accuracy, correct_predictions


	# Define a function to train the neural network using a data loader as input
	def train(self, data_loader, eta, stopping_criterion = 'Epochs', num_epochs = 1000, is_verbose = True) :

		"""
		inputs :

		data_loader : 	
			A class instance of the dataLoader class, containing the standard utilities for next batch and split data recovery
		eta :
			The base learning rate
		stopping_criterion : 'Epochs'
			The criterion for stopping the training of the neural network
		num_epochs : 1000
			The passes over the entire dataset
		is_verbose : True
			Whether to print the information
		"""

		"""
		outputs :

		tr_loss_list :
			The list containing the loss per iteration for all epochs
		tr_acc_list :
			The list containing the accuracy per iteration for all epochs
		val_loss_list :
			The list of loss on validation set
		val_acc_list :
			The list of accuracy on validation set
		"""

		is_continue_training = True
		
		tr_loss_list = []
		tr_acc_list = []
		val_loss_list = []
		val_acc_list = []

		# Set the stopping criterion
		if stopping_criterion == 'Epochs' :
			threshold_stop_training = num_epochs
			trigger_stop_training = 0

		# Training loop
		while is_continue_training :

			iteration_count = 0

			# For each epoch ...
			data_loader.ResetDataSplit(split = 'Train')
			iteration_tr_acc_list = []
			iteration_tr_loss_list = []

			while data_loader.IsNextBatchExists(split = 'Train') :
				
				iteration_count += 1
				# Get the next batch
				x_tr_batch, y_tr_batch = data_loader.GetNextBatch(split = 'Train')
				# Run forward pass
				pred_batch = self.forward(np.transpose(x_tr_batch))
				# Calculate the loss
				loss_batch = self.loss(y_true = y_tr_batch)
				# Run backward pass
				self.backward(y_true = y_tr_batch)
				# Update the weights
				self.update(eta)
				# Calculate the accuracy
				acc_batch, corr_batch = self.EvaluateAccuracy(y_true = y_tr_batch)
				# Store in the loss
				iteration_tr_acc_list.append(acc_batch)
				iteration_tr_loss_list.append(loss_batch)

				# if is_verbose :
				# 	print('[DEBUG] Epoch :\t', trigger_stop_training + 1, '\t\tIteration : ', iteration_count)

			tr_loss_list.append(iteration_tr_loss_list)
			tr_acc_list.append(iteration_tr_acc_list)

			# After each epoch, calculate the validation performance
			x_val_batch, y_val_batch = data_loader.GetDataSplit('Test')
			pred_val = self.forward(np.transpose(x_val_batch))
			loss_val = self.loss(y_true = y_val_batch)
			acc_val, corr_val = self.EvaluateAccuracy(y_true = y_val_batch)
			val_acc_list.append(acc_val)
			val_loss_list.append(loss_val)

			# Verbose
			if is_verbose :
				print('[INFO] Epoch :\t', trigger_stop_training + 1)
				print('[INFO] Training Loss :\t', loss_batch, '\tTraining Acc :\t', acc_batch)
				print('[INFO] Validation Loss :\t', loss_val, '\tValidation Acc :\t', acc_val)

			# Re-evaluate the criterion for stopping
			if stopping_criterion == 'Epochs' :
				trigger_stop_training += 1
				if trigger_stop_training >= threshold_stop_training :
					is_continue_training = False
				else :
					is_continue_training = True

		# Return all the stuff
		return tr_loss_list, tr_acc_list, val_loss_list, val_acc_list


	# Define a function to test the neural network using a data loader as input
	def test(self, data_loader, split = 'Test', x_def = None, y_def = None, is_verbose = True) :

		"""
		inputs :

		data_loader : 	
			A class instance of the dataLoader class, containing the standard utilities for next batch and split data recovery
		split : 'Test'
			The split in which we want to check the performance of the neural network. SUPPORT : 'Train', 'Validation', 'Test', 'Default'
		is_verbose : True
			Whether to display the information
		"""

		"""
		outputs :

		loss :
			The loss on the split
		acc :
			The accuracy on the split
		corr :
			The correct predictions
		"""

		if split == 'Train' :
			x_test, y_test = data_loader.GetDataSplit('Train')
		elif split == 'Validation' :
			x_test, y_test = data_loader.GetDataSplit('Validation')
		elif split == 'Test' :
			x_test, y_test = data_loader.GetDataSplit('Test')
		elif split == 'Default' :
			x_test = x_def
			y_test = y_def
		# No code should ever reach this place
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)

		# Get prediction
		pred = self.forward(np.transpose(x_test))
		# Get loss
		loss = self.loss(y_true = y_test)
		# Evaluate the accuracy
		acc, corr = self.EvaluateAccuracy(y_true = y_test)

		return loss, acc, corr


	# Define a method for saving weights upon command, storing the weights and biases
	def SaveModel(self, path) :

		"""
		inputs :

		path :
			The path to which the weights need to be saved
		"""

		"""
		outputs :
		"""

		# Create a dict of weights and biases
		weights_saver_dict = {}
		biases_saver_dict = {}

		for i in self.weights_dict :
			weights_saver_dict[i] = np.copy(self.weights_dict[i])
		for i in self.biases_dict :
			biases_saver_dict[i] = np.copy(self.biases_dict[i])

		# Remove if previous snapshot
		os.system('rm -f ' + str(os.path.join(path, 'weights_dict.pkl')) + '   ' + os.path.join(path, 'biases_dict.pkl'))

		# Dump to pickle
		with open(os.path.join(path, 'weights_dict.pkl'), 'wb') as weights_saver :
			pickle.dump(weights_saver_dict, weights_saver, protocol = pickle.HIGHEST_PROTOCOL)
		with open(os.path.join(path, 'biases_dict.pkl'), 'wb') as biases_saver :
			pickle.dump(biases_saver_dict, biases_saver, protocol = pickle.HIGHEST_PROTOCOL)


	# Define a method for loading saved weights upon command
	def LoadModel(self, path) :

		"""
		inputs :

		path :
			The path to which the weights need to be saved
		"""

		"""
		outputs :
		"""

		# Load from pickle
		with open(os.path.join(path, 'weights_dict.pkl'), 'rb') as weights_loader :
			weights_loader_dict = pickle.load(weights_loader)
		with open(os.path.join(path, 'biases_dict.pkl'), 'rb') as biases_loader :
			biases_loader_dict = pickle.load(biases_loader)

		for i in weights_loader_dict :
			self.weights_dict[i] = np.copy(weights_loader_dict[i])
		for i in biases_loader_dict :
			self.biases_dict[i] = np.copy(biases_loader_dict[i])


	# Define a method to compute numerical gradients with respect to parameters of a particular layer
	def ComputeNumericalGradients(self, x, y, parameter, perturb_loc, perturb_amount) :

		"""
		inputs :

		x :
			The input on which numerical gradient needs to be evaluated
		y :
			The ground truth corresponding to the input data
		parameter :
			The parameter for which the numerical gradient needs to be evaluated. EX : 'W1', 'b2' etc.
		perturb_loc :
			The location in the weight/bias where we need to perturb
		perturb_amount : 
			The magnitude with which to perturb
		"""

		"""
		outputs :

		left_deriv :
			The left-sided derivative (f(x) - f(x - eps))/(eps)
		right_deriv :
			The right-sided derivative (f(x + eps) - f(x))/(eps)
		center_deriv :
			The centered derivative (f(x + eps) - f(x - eps))/(2*eps)
		"""

		# print('[DEBUG] Parameter : ', parameter)
		# for i in self.weights_dict :
		# 	print('[DEBUG] Weights : ', i)
		# for i in self.biases_dict :
		# 	print('[DEBUG] Biases : ', i)

		# print('[DEBUG] True Label : ', y)

		# Save the current param
		if 'W' in parameter :
			current_param = np.copy(self.weights_dict[parameter])
		elif 'b' in parameter :
			current_param = np.copy(self.biases_dict[parameter])
		# No code should ever reach here ...
		else :
			print('[ERROR] Wrong parameter enquired : ', str(parameter))
			print('[ERROR] Terminating the code ...')
			sys.exit()

		# Get the perturbation
		perturbation = np.zeros_like(current_param).astype(np.float32)
		perturbation[perturb_loc[0], perturb_loc[1]] = np.abs(perturb_amount)
		
		# print('[DEBUG] Perturbation : ', (perturbation == 0).all())
		# print('[DEBUG] Perturbation[i, j] : ', perturbation[perturb_loc[0], perturb_loc[1]])
		# print('[DEBUG] Perturbation : ', (perturbation == 0).all())
		# print('[DEBUG] Perturbation Max : ', np.max(perturbation))

		pos_perturbed_param = current_param + perturbation
		neg_perturbed_param = current_param - perturbation

		# Get the loss with the parameter
		pred_center = self.forward(x = x)
		loss_center = self.loss(y_true = y, y_pred = pred_center)
		# print('[DEBUG] Center Prediction : ', pred_center)
		# print('[DEBUG] Center Loss : ', loss_center)

		# Set the parameter to +ve perturbation
		if 'W' in parameter :
			self.weights_dict[parameter] = np.copy(pos_perturbed_param)
		elif 'b' in parameter :
			self.biases_dict[parameter] = np.copy(pos_perturbed_param)
		# No code should ever reach here ...
		else :
			print('[ERROR] Wrong parameter enquired : ', str(parameter))
			print('[ERROR] Terminating the code ...')
			sys.exit()
		# Get the +ve perturbation loss
		pred_right = self.forward(x = x)
		loss_right = self.loss(y_true = y, y_pred = pred_right)
		# print('[DEBUG] Right Prediction : ', pred_right)
		# print('[DEBUG] Right Loss : ', loss_right)

		# Set the parameter to -ve perturbation
		if 'W' in parameter :
			self.weights_dict[parameter] = np.copy(neg_perturbed_param)
		elif 'b' in parameter :
			self.biases_dict[parameter] = np.copy(neg_perturbed_param)
		# No code should ever reach here ...
		else :
			print('[ERROR] Wrong parameter enquired : ', str(parameter))
			print('[ERROR] Terminating the code ...')
			sys.exit()

		# Get the +ve perturbation loss
		pred_left = self.forward(x = x)
		loss_left = self.loss(y_true = y, y_pred = pred_left)
		# print('[DEBUG] Left Prediction : ', pred_left)
		# print('[DEBUG] Left Loss : ', loss_left)
		
		# print('[DEBUG] Predictions Identical ? : ', (pred_center == pred_left).all())
		# print('[DEBUG] Predictions Identical ? : ', (pred_center == pred_right).all())
		# print('[DEBUG] Predictions Identical ? : ', (pred_left == pred_right).all())

		# Compute the left-sided, right-sided and centered derivative
		left_deriv = (loss_center - loss_left)*1.0/perturb_amount
		right_deriv = (loss_right - loss_center)*1.0/perturb_amount
		center_deriv = (loss_right - loss_left)*1.0/(2.0*perturb_amount)

		# Restore the parametes
		if 'W' in parameter :
			self.weights_dict[parameter] = np.copy(current_param)
		elif 'b' in parameter :
			self.biases_dict[parameter] = np.copy(current_param)
		# No code should ever reach here ...
		else :
			print('[ERROR] Wrong parameter enquired : ', str(parameter))
			print('[ERROR] Terminating the code ...')
			sys.exit()

		# Return!
		return left_deriv, right_deriv, center_deriv


	# Define a methd that computes the parameters of the architecture
	def GetParameterNumber(self) :

		"""
		inputs :
		"""

		"""
		outputs :

		param_count :
			The net number of parameters
		"""

		param_count = 0

		# Process each weight
		for i in self.weights_dict :
			a_param_shape = list(self.weights_dict[i].shape)
			print('[DEBUG] Current Shape : ', a_param_shape)
			a_param_count = 1
			for j in a_param_shape :
				a_param_count *= j
			param_count += a_param_count
		# Process each bias
		for i in self.biases_dict :
			a_param_shape = list(self.biases_dict[i].shape)
			print('[DEBUG] Current Shape : ', a_param_shape)
			a_param_count = 1
			for j in a_param_shape :
				a_param_count *= j
			param_count += a_param_count

		# Return 
		return param_count


	# Override the method to print the neural network
	def __repr__(self) :

		"""
		inputs :
		"""

		"""
		outputs :

		string (implicit) :
			The string representation
		"""

		self.__str__()


	# Override the method to convert the neural network into string
	def __str__(self) :

		"""
		inputs :
		"""

		"""
		outputs :

		string (implicit) :
			The string representation
		"""

		# Print the description of the layers
		string = 'Neural Network ('
		for i in range(len(self.weights_dict)) :
			string += '\n\t' 
			string += 'Layer('
			string += 'W' + str(i + 1) + ' : '
			string += str(list(self.weights_dict['W' + str(i + 1)].shape))
			string += '\t,\t'
			string += 'b' + str(i + 1) + ' : '
			string += str(list(self.biases_dict['b' + str(i + 1)].shape))
			string += '\t,\t'
			string += self.non_linearity_dict['g' + str(i + 1)][0] + '\t:\t'
			string += str([self.non_linearity_dict['g' + str(i + 1)][1], self.non_linearity_dict['g' + str(i + 1)][1]])
			string += ')'
		string += '\n)'

		return string


 # Psuedo-main
if __name__ == '__main__' :

	# Create a class instance
	print('##################################################')
	print('########## TEST : Neural Network')
	print('##################################################')
	model = multiLayerPerceptron()
	print(model)

	# Initialize weights
	print('##################################################')
	print('########## TEST : Initializers')
	print('##################################################')
	model.initialize_weights('Zero')
	print(model)
	for i in range(len(model.weights_dict)) :
		w = model.GetWeight(i + 1)
		print('[DEBUG] Zero Initialization for Weights Successful : ', np.all(w == 0.0))
		b = model.GetBias(i + 1)
		print('[DEBUG] Zero Initialization for Biases Successful : ', np.all(b == 0.0))
	model.initialize_weights('Normal')
	print(model)
	for i in range(len(model.weights_dict)) :
		w = model.GetWeight(i + 1)
		print('[DEBUG] Empirical Mean of Weights : ', np.mean(w), ' , Empirical Std of Weights : ', np.std(w), '\t (N(0, 1))')
		b = model.GetBias(i + 1)
		print('[DEBUG] Zero Initialization for Biases Successful : ', np.all(b == 0.0))
	model.initialize_weights('Glorot')
	print(model)
	for i in range(len(model.weights_dict)) :
		w = model.GetWeight(i + 1)
		print('[DEBUG] Empirical Mean of Weights : ', np.mean(w), ' (Theoretical : 0)', ', Empirical Std of Weights : ', np.std(w), ' (Thoretical : ', np.sqrt((2*np.sqrt(6.0/(w.shape[0] + w.shape[1])))**2/12), ')')
		b = model.GetBias(i + 1)
		print('[DEBUG] Zero Initialization for Biases Successful : ', np.all(b == 0.0))

	# Test the forward
	print('##################################################')
	print('########## TEST : Forward')
	print('##################################################')
	x_batch = np.random.random([784, 4]) # A batch input
	pred = model.forward(x_batch)
	for i in model.preactivation_dict :
		if model.preactivation_dict[i] is not None :
			print('[DEBUG] The Pre-Activation : ' + str(i) + ', Shape : ' + str(model.preactivation_dict[i].shape))
	for i in model.activation_dict :
		if model.activation_dict[i] is not None :
			print('[DEBUG] The Activation : ' + str(i) + ', Shape : ' + str(model.activation_dict[i].shape))
	print(pred)
	print(np.sum(pred, 0))

	# Test the backward
	print('##################################################')
	print('########## TEST : Backward')
	print('##################################################')
	x_batch = np.random.random([784, 4]) # A batch input
	y_batch = np.random.randint(0, 2, [4,]).astype(np.float32)
	print('[DEBUG] True Labels : ', y_batch)
	pred = model.forward(x_batch)
	model.backward(y_batch)
	
	print('##########')
	for i in model.preactivation_dict :
		if model.preactivation_dict[i] is not None :
			print('[DEBUG] The Pre-Activation : ' + str(i) + ', Shape : ' + str(model.preactivation_dict[i].shape))
	for i in model.grad_preactivation_dict :
		if model.grad_preactivation_dict[i] is not None :
			print('[DEBUG] The Gradient of Pre-Activation : ' + str(i) + ', Shape : ' + str(model.grad_preactivation_dict[i].shape))
	print('##########')
	for i in model.activation_dict :
		if model.activation_dict[i] is not None :
			print('[DEBUG] The Activation : ' + str(i) + ', Shape : ' + str(model.activation_dict[i].shape))
	for i in model.grad_activation_dict :
		if model.grad_activation_dict[i] is not None :
			print('[DEBUG] The Gradient of Activation : ' + str(i) + ', Shape : ' + str(model.grad_activation_dict[i].shape))
	print('##########')
	for i in model.weights_dict :
		if model.weights_dict[i] is not None :
			print('[DEBUG] The Weight : ' + str(i) + ', Shape : ' + str(model.weights_dict[i].shape))
	for i in model.grad_weights_dict :
		if model.grad_weights_dict[i] is not None :
			print('[DEBUG] The Gradient of Weight : ' + str(i) + ', Shape : ' + str(model.grad_weights_dict[i].shape))
	print('##########')
	for i in model.biases_dict :
		if model.biases_dict[i] is not None :
			print('[DEBUG] The Bias : ' + str(i) + ', Shape : ' + str(model.biases_dict[i].shape))
	for i in model.grad_biases_dict :
		if model.grad_biases_dict[i] is not None :
			print('[DEBUG] The Gradient of Bias : ' + str(i) + ', Shape : ' + str(model.grad_biases_dict[i].shape))
	print('##########')
	print(pred)
	print(np.sum(pred, 0))

	# Test the loss
	print('##################################################')
	print('########## TEST : Loss')
	print('##################################################')
	x_batch = np.random.random([784, 4]) # A batch input
	y_batch = np.random.randint(0, 2, [4,]).astype(np.float32)
	print('[DEBUG] True Labels : ', y_batch)
	print(pred)
	print(model.loss(y_true = y_batch)) # It matches the desired thing correctly

	# Test the update
	print('##################################################')
	print('########## TEST : Update')
	print('##################################################')
	model = multiLayerPerceptron(activation_list = ('Tanh', 'Tanh'))
	model.initialize_weights('Glorot')
	x_batch = np.random.random([784, 4]) # A batch input
	y_batch = np.random.randint(0, 2, [4,]).astype(np.float32)
	print('[DEBUG] True Labels : ', y_batch)
	for i in range(10) :
		pred = model.forward(x_batch)
		print('[DEBUG]\t\tGradient Descent Iteration :\t', i, '\t\tLoss : ', model.loss(y_batch))
		model.backward(y_batch)
		model.update(eta = 0.001)

	# Test the accuracy
	print('##################################################')
	print('########## TEST : Accuracy')
	print('##################################################')
	model = multiLayerPerceptron(activation_list = ('ReLU', 'ReLU'))
	model.initialize_weights('Glorot')
	x_batch = np.random.random([784, 40]) # A batch input
	y_batch = np.random.randint(0, 2, [40,]).astype(np.float32)
	pred = model.forward(x_batch)
	acc_1, cor_1 = model.EvaluateAccuracy(y_true = y_batch)
	acc_2, cor_2 = model.EvaluateAccuracy(y_true = y_batch, y_pred = pred)
	print('[DEBUG] Predictions by both methods match : ' + str(np.all(acc_1 == acc_2)))
	print('[DEBUG] Predictions by both methods match : ' + str(np.all(cor_1 == cor_2)))
	print('[DEBUG] Initial Accuracy : ', acc_1)
	for i in range(10) :
		pred = model.forward(x_batch)
		print('[DEBUG]\t\tGradient Descent Iteration :\t', i, '\t\tLoss : ', model.loss(y_batch))
		model.backward(y_batch)
		model.update(eta = 0.001)
	acc_1, cor_1 = model.EvaluateAccuracy(y_true = y_batch)
	print('[DEBUG] Final Accuracy : ', acc_1)

	# Test the training script
	print('##################################################')
	print('########## TEST : Train')
	print('##################################################')
	num_epochs_ = 50
	mnist = utils.dataLoader(batch_size = 5000)
	model = multiLayerPerceptron(n_hidden = 2, hidden_dims = (750, 500), activation_list = ('Sigmoid', 'Sigmoid'))

	# Test the parameter count
	print('##################################################')
	print('########## TEST : Parameter Count')
	print('##################################################')
	print('[DEBUG] Parameter count : ', model.GetParameterNumber())

	model.initialize_weights('Glorot')
	time_1 = time.time()
	tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = model.train(data_loader = mnist, eta = 1, num_epochs = num_epochs_, is_verbose = True)
	time_2 = time.time()
	print('[DEBUG] Time Per Epoch : ', (time_2 - time_1)/num_epochs_)
	te_loss, te_acc, te_corr = model.test(data_loader = mnist, split = 'Test', x_def = None, y_def = None, is_verbose = True)
	print('[DEBUG] Testing Loss\t:\t', te_loss, '\tAccuracy\t:\t', te_acc)
	te_loss, te_acc, te_corr = model.test(data_loader = mnist, split = 'Train', x_def = None, y_def = None, is_verbose = True)
	print('[DEBUG] Testing Loss\t:\t', te_loss, '\tAccuracy\t:\t', te_acc)

	# # Test the numerical gradient computation
	# print('##################################################')
	# print('########## TEST : Numerical Gradients')
	# print('##################################################')
	# mnist = utils.dataLoader(batch_size = 1)
	# parameter = 'b2'
	# perturb_loc = (0, 0)
	# x_batch, y_batch = mnist.GetNextBatch()
	# left_grad, right_grad, center_grad = model.ComputeNumericalGradients(x = np.transpose( x_batch ), y = y_batch, parameter = parameter, perturb_loc = perturb_loc, perturb_amount = 1e-5)
	# print('[DEBUG] Left-Sided Gradient : ', left_grad)
	# print('[DEBUG] Right-Sided Gradient : ', right_grad)
	# print('[DEBUG] Centered Gradient : ', center_grad)
	# y_pred = model.forward(x = np.transpose(x_batch))
	# model.backward(y_true = y_batch) 
	# if 'W' in parameter :
	# 	true_grad = model.grad_weights_dict['grad_' + parameter][perturb_loc[0], perturb_loc[1]]
	# elif 'b' in parameter :
	# 	true_grad = model.grad_biases_dict['grad_' + parameter][perturb_loc[0], perturb_loc[1]]
	# print('[DEBUG] True Gradient : ', true_grad)

	# # Test the save and load properties
	# print('##################################################')
	# print('########## TEST : Save and Load Weights')
	# print('##################################################')
	# model.SaveModel(path = './')
	# model_new = multiLayerPerceptron(n_hidden = 2, hidden_dims = (500, 300), activation_list = ('ReLU', 'ReLU'))
	# model_new.LoadModel(path = './')
	# te_loss, te_acc, te_corr = model_new.test(data_loader = mnist, split = 'Test', x_def = None, y_def = None, is_verbose = True)
	# print('[DEBUG] Testing Loaded Loss\t:\t', te_loss, '\tAccuracy\t:\t', te_acc)
	# te_loss, te_acc, te_corr = model_new.test(data_loader = mnist, split = 'Train', x_def = None, y_def = None, is_verbose = True)
	# print('[DEBUG] Testing Loaded Loss\t:\t', te_loss, '\tAccuracy\t:\t', te_acc)