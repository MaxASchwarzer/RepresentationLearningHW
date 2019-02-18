# Dependencies
import sys

import numpy as np
import matplotlib.pyplot as plt

from Neural_Network import multiLayerPerceptron
import Utilities as utils
import Plot_Utilities as plt_utils

# Pseudo-main
if __name__ == '__main__' :

	N_HIDDEN = 2
	# Fine-Tuned Architectures
	HIDDEN_DIMS = (500, 300)
	ACTIVATION_LIST = ('ReLU', 'ReLU')
	LEARNING_RATE = 0.5
	NUM_EPOCHS = 100
	WEIGHT_INITIALIZER = 'Glorot'
	BATCH_SIZE = 5000
	TRAIN = False

	# Create a model
	model = multiLayerPerceptron(n_hidden = N_HIDDEN, hidden_dims = HIDDEN_DIMS, activation_list = ACTIVATION_LIST)
	# Initialize weights
	model.initialize_weights(initializer = WEIGHT_INITIALIZER)

	# Create mnist loader
	mnist = utils.dataLoader(batch_size = BATCH_SIZE)

	# Train the architecture
	if TRAIN :
		_, _, _, _ = model.train(data_loader = mnist, eta = LEARNING_RATE, num_epochs = NUM_EPOCHS)
		# Save the weights, these are the best weights!
		model.SaveModel(path = './')
	else :
		model.LoadModel(path = './')
	

	# Create a list of perturbation locations. This is hand-coded for the purpose of the assignment
	perturbation_loc = [(0, loc) for loc in range(10)]
	perturbation_param = 'W2' # As per the question

	# Create a new loader and get the fixed data point
	mnist_temp = utils.dataLoader(batch_size =  1)
	x_single, y_single = mnist_temp.GetNextBatch(split = 'Train')

	# Select the epsilon
	I_LIST = [0.0, 1.0]
	K_LIST = [1.0, 2.0, 3.0, 4.0, 5.0]

	epsilon_expt_list = []
	max_num_grad_dev_list = []

	for i in I_LIST :
		for k in K_LIST :
	
			# Create epsilon
			epsilon = 1.0/(np.power(10, i)*k)

			# Create a list of current deviations
			num_grad_dev_list = []

			# For each of the perturbation location :
			for a_perturbation_loc in perturbation_loc :
				
				# Get the numerically estimated loss
				left_grad, right_grad, center_grad = model.ComputeNumericalGradients(	x = np.transpose( x_single ), 
																						y = y_single, 
																						parameter = perturbation_param, 
																						perturb_loc = a_perturbation_loc, 
																						perturb_amount = epsilon )

				# Get exact gradient
				y_pred = model.forward(x = np.transpose(x_single))
				model.backward(y_true = y_single) 
				if 'W' in perturbation_param :
					true_grad = model.grad_weights_dict['grad_' + perturbation_param][a_perturbation_loc[0], a_perturbation_loc[1]]
				elif 'b' in perturbation_param :
					true_grad = model.grad_biases_dict['grad_' + perturbation_param][a_perturbation_loc[0], a_perturbation_loc[1]]
				
				num_grad_dev_list.append(np.abs(center_grad - true_grad))

			# Add the record
			epsilon_expt_list.append(epsilon)
			max_num_grad_dev_list.append(np.max(np.array(num_grad_dev_list)))

			print('[DEBUG] Epsilon : ', epsilon, ' Maximum Difference : ', np.max(np.array(num_grad_dev_list)))

	# Plot the graph
	fig = plt.figure()
	# axis = fig.gca()

	x_data = [np.log10(an_eps_log) for an_eps_log in epsilon_expt_list]

	plt_utils.PlotScatter(x_ = x_data, y_ = max_num_grad_dev_list, c_ = '#CC4F1B', marker_ = 'x', label_ = 'Maximum Absolute Difference between Numerical and Actual Grad', alpha_ = 0.75, ms_ = None)
	plt_utils.PlotFunction(x_ = x_data, y_ = max_num_grad_dev_list, c_ = '#CC4F1B', label_ = 'Plot of Maximum Absolute Difference between Numerical and Actual Grad ', alpha_ = 0.75)

	plt.xlabel(r'$\log_{10}(\epsilon) \rightarrow$')
	plt.ylabel(r'$\max_{1\leq i\leq 10}\mid\nabla_i^N - \frac{\partial L}{\partial \theta_i}\mid\rightarrow$')

	plt.title('Comparison between Maximum Absolute Difference between Numerical and Actual Grad')

	plt.legend(loc = 'upper left')

	y_max = np.max(np.array(max_num_grad_dev_list))
	y_min = np.min(np.array(max_num_grad_dev_list))
	plt.ylim((y_min, y_max))

	# plt.autoscale(enable = True, axis = 'y')

	plt.show()
