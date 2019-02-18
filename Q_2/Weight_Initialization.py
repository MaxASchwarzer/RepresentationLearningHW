# Dependencies
import numpy as np
import os, sys
import time
import pickle

import Utilities as utils
from Neural_Network import multiLayerPerceptron

# Psuedo-main
if __name__ == '__main__' :

	# Global parameters
	NUM_RUNS = 10
	
	N_HIDDEN = 2
	HIDDEN_DIMS = (500, 300)
	ACTIVATION_LIST = ('ReLU', 'ReLU')
	BATCH_SIZE = 5000
	LEARNING_RATE = 0.05
	NUM_EPOCHS = 10

	# Weight initializers
	weight_initializers_list = ['Zero', 'Normal', 'Glorot']

	# Run the code for each of the initializers
	for an_initializer in weight_initializers_list :

		print('####################################################################################################')
		print('[INFO] Initializer : ', an_initializer)
		print('####################################################################################################')

		# Create a list to hold the list of lists of train performances as required
		list_of_train_perf_list = []

		# Create a model 
		model = multiLayerPerceptron(n_hidden = N_HIDDEN, hidden_dims = HIDDEN_DIMS, activation_list = ACTIVATION_LIST, mode = 'train', datapath = None, model_path = None)
		print('[INFO] The MLP Architecture : ')
		print(model)
		print('[INFO] The number of parameters : \n', model.GetParameterNumber())


		num_runs = NUM_RUNS
		# For each of the runs
		for a_run in range(num_runs) :

			print('[INFO] Run : ', a_run + 1)

			# Create a list to hold the current run train, val and test losses 
			train_perf_run_list = []

			# Initialize weights
			model = multiLayerPerceptron(n_hidden = N_HIDDEN, hidden_dims = HIDDEN_DIMS, activation_list = ACTIVATION_LIST, mode = 'train', datapath = None, model_path = None)
			model.initialize_weights(initializer = an_initializer, hidden_dims = HIDDEN_DIMS)	

			# Create the dataset loader
			mnist = utils.dataLoader(batch_size = BATCH_SIZE)

			# Get pre-training performance as the ground reference
			loss, acc, corr = model.test(data_loader = mnist, split = 'Train')
			print('[INFO] EPOCH\t:\t0\t\tLoss\t:\t', loss, '\t\tAccuracy\t:\t', acc)
			train_perf_run_list.append(loss)

			# Epochs 
			for an_epoch in range(NUM_EPOCHS) :

				# Train
				_, _, _, _ = model.train(data_loader = mnist, eta = LEARNING_RATE, stopping_criterion = 'Epochs', num_epochs = 1, is_verbose = False)
				loss, acc, corr = model.test(data_loader = mnist, split = 'Train', is_verbose = False)
				print('[INFO] EPOCH\t:\t', an_epoch + 1, '\t\tLoss\t:\t', loss, '\t\tAccuracy\t:\t', acc)	
				train_perf_run_list.append(loss)

			# Append globally
			list_of_train_perf_list.append(train_perf_run_list)

		# Store the final list of lists as numpy array
		list_of_train_perf_list = np.array(list_of_train_perf_list).astype(np.float32)
		print('[DEBUG] Shape of Resultant Performance Matrix : ', list_of_train_perf_list.shape)
		np.save('Weight_Initialization_' + an_initializer, list_of_train_perf_list)

