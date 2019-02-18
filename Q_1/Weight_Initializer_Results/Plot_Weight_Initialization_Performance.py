# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import Plot_Utilities as plt_utils


# Pseudo-main :
if __name__ == '__main__' :

	# Load the files
	zero_init = np.load('Weight_Initialization_Zero.npy')
	normal_init = np.load('Weight_Initialization_Normal.npy')
	glorot_init = np.load('Weight_Initialization_Glorot.npy')

	# Sanity
	print('[DEBUG] Shape Zero Init : ', zero_init.shape)
	print('[DEBUG] Shape Normal Init : ', normal_init.shape)
	print('[DEBUG] Shape Glorot Init : ', glorot_init.shape)

	# Get the means and errors
	zero_mean = np.mean(zero_init, axis = 0)
	zero_std = np.std(zero_init, axis = 0)
	print('[DEBUG] Zero Mean : ', zero_mean, ' Zero Std : ', zero_std)

	normal_mean = np.mean(normal_init, axis = 0)
	normal_std = np.std(normal_init, axis = 0)
	print('[DEBUG] Normal Mean : ', normal_mean, ' Normal Std : ', normal_std)

	glorot_mean = np.mean(glorot_init, axis = 0)
	glorot_std = np.std(glorot_init, axis = 0)
	print('[DEBUG] Glorot Mean : ', glorot_mean, ' Glorot Std : ', glorot_std)

	# Plot the graphs
	fig = plt.figure()
	axis = fig.gca()

	x_data = np.arange(0, 11)

	# Plot zero and shade the region
	plt_utils.PlotScatter(x_ = x_data, y_ = zero_mean, c_ = '#CC4F1B', marker_ = 'x', label_ = 'Zero Initialization Loss Points', alpha_ = 0.75, ms_ = None)
	plt_utils.PlotFunction(x_ = x_data, y_ = zero_mean, c_ = '#CC4F1B', label_ = 'Zero Initialization Loss Mean', alpha_ = 0.75)
	plt.fill_between(x_data, zero_mean - zero_std, zero_mean + zero_std, alpha = 0.25, facecolor='#FF9848', label = 'Zero Initialization Loss Std')

	plt_utils.PlotScatter(x_ = x_data, y_ = normal_mean, c_ = '#1B2ACC', marker_ = 'x', label_ = 'Normal Initialization Loss Points', alpha_ = 0.75, ms_ = None)
	plt_utils.PlotFunction(x_ = x_data, y_ = normal_mean, c_ = '#1B2ACC', label_ = 'Normal Initialization Error Mean', alpha_ = 0.75)
	plt.fill_between(x_data, normal_mean - normal_std, normal_mean + normal_std, alpha = 0.25, facecolor='#1B2ACC', label = 'Normal Initialization Loss Std')

	plt_utils.PlotScatter(x_ = x_data, y_ = glorot_mean, c_ = '#3F7F4C', marker_ = 'x', label_ = 'Glorot Initialization Loss Points', alpha_ = 0.75, ms_ = None)
	plt_utils.PlotFunction(x_ = x_data, y_ = glorot_mean, c_ = '#3F7F4C', label_ = 'Glorot Initialization Error Mean', alpha_ = 0.75)
	plt.fill_between(x_data, glorot_mean - glorot_std, glorot_mean + glorot_std, alpha = 0.25, facecolor='#3F7F4C', label = 'Glorot Initialization Loss Std')

	plt.xlabel(r'Number of Epochs $\rightarrow$')
	plt.ylabel(r'Averaged Training Loss $\rightarrow$')

	plt.title('Graph of Average Training Loss as a Function of Different Weight Initializations')

	plt.legend(loc = 'upper right')

	plt.grid()
	axis.set_xticks(np.arange(0, 11, 1))
	axis.set_yticks(np.arange(0, 22, 1))

	plt.show()