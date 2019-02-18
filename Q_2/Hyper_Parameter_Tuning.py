# Dependencies
import numpy as np
import os, sys
import time
import pickle

import Utilities as utils
from Neural_Network import multiLayerPerceptron

# Psuedo-main
if __name__ == '__main__' :

	# Grid Search Parameters
	HIDDEN_DIM_LIST = [(500, 300), (675, 400), (750, 500)]
	NON_LINEARITY_LIST = [('ReLU', 'ReLU'), ('Sigmoid', 'Sigmoid'), ('Tanh', 'Tanh')]
	LEARNING_RATE_LIST = [1, 1e-1, 1e-2]

	# Global Parameters
	NUM_EPOCHS = 1
	BATCH_SIZE = 5000
	N_HIDDEN = 2

	# File for storing the results
	fp = open('./Hyper_Parameter_Tuning.txt', 'w')

	# Grid-search
	for a_hidden_dim in HIDDEN_DIM_LIST :
		for a_non_lin in NON_LINEARITY_LIST :
			for a_learning_rate in LEARNING_RATE_LIST :

				# Create a model
				model = multiLayerPerceptron(n_hidden = N_HIDDEN, hidden_dims = a_hidden_dim, activation_list = a_non_lin, mode = 'train', datapath = None, model_path = None)
				model.initialize_weights(initializer = 'Glorot', hidden_dims = a_hidden_dim)	
				# Create new dataloader
				mnist = utils.dataLoader(batch_size = BATCH_SIZE)

				# Train the model
				_, _, _, _ = model.train(data_loader = mnist, eta = a_learning_rate, stopping_criterion = 'Epochs', num_epochs = NUM_EPOCHS, is_verbose = True)
				loss, acc, corr = model.test(data_loader = mnist, split = 'Validation', is_verbose = False)
				string = '\n' + 'Hidden_Dimensions\t:\t' + str(a_hidden_dim) + '\tNon_Linearities\t:\t' + str(a_non_lin) + '\tLearning_Rate\t:\t' + str(a_learning_rate) + '\tEpochs\t:\t' + str(NUM_EPOCHS) + '\tBatch\t:\t' + str(BATCH_SIZE) + '\tLoss\t:\t' + str(loss) + '\tAccuracy\t:\t' + str(acc)
				print(string)
				fp.write(string)

	# Close file
	fp.close()