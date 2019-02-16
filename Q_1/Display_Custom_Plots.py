# Dependencies
import os
import sys
import matplotlib.pyplot as plt
# plt.ion() # NOT WORKING PROPERLY!!
import matplotlib
# matplotlib.interactive(True) # Set interactive mode # NOT WORKING PROPERLY!!
# matplotlib.use("TkAgg") # Set interactive mode # NOT WORKING PROPERLY!!
import cv2
import numpy as np
from Print_Updating_Info import screenToPrintUpdatingInfo
import time


# Define a class to hold all the methods to create custom plots
class screenToDisplayCustomPlots(object) :

	"""
	attributes--

	__init__(self, is_close_prev_figs = True) :
		The constructor		
	"""


	# Define the constructor
	def __init__(self, is_close_prev_figs = True) :

		"""
		inputs--

		is_close_prev_figs : True
			Whether to close all previous figures
		"""

		# Close all previous figures, if instructed to do so
		if is_close_prev_figs :
			plt.close('all')


	# Define a function to display the images at input paths/list entries/numpy arrays in an array
	def DisplayImagesFromSourcesInCustomPlot(self, print_screen, image_sources, labels_true = None, labels_pred = None, plot_dim_1 = -1, plot_dim_2 = -1, interp = 'spline16', is_close_prev_figs = True, is_rescale_if_out_of_range = True, is_timer_close = True, timer_close = 5) :

		"""
		inputs--

		print_screen :
			The screen onto which the information is to be printed. An instance of 
		image_sources :
			A list of image paths
		labels_true : None
			Ground truth labels. Not to be printed if None
		labels_pred : None
			Predicted labels. Not to be printed if None
		plot_dim_1 : -1
			Number of rows in the plot. To be calculated appropriately if -1
		plot_dim_2 : -1
			Number of rows in the plot. To be calculated appropriately if -1
		interp : 'spline16'
			The interpolation method for displaying images. Allowed entries are 'spline16' and 'nearest'
		is_close_prev_figs : True
			Whether to close all previous figures
		is_rescale_if_out_of_range : True
			Whether to rescale the values should the maximum of the image is <= 1.0
		is_timer_close : True
			Whether to close the figure based on a timer
		"""

		"""
		outputs--
		"""

		if is_close_prev_figs :
			plt.close('all')

		# Infer type of image sources
		if isinstance(image_sources, list) :
			len_image_sources = len(image_sources)
			image_sources_type = 'list'
		elif isinstance(image_sources, tuple) :
			len_image_sources = len(image_sources)
			image_sources_type = 'tuple'
		elif isinstance(image_sources, np.ndarray) :
			len_image_sources = image_sources.shape[0]
			image_sources_type = 'np.ndarray'
		else :
			print_screen.PrintStaticInfo('[ERROR] Unsupported input format for image sources.')
			print_screen.PrintStaticInfo('[ERROR] Terminating the program ...')
			sys.exit()

		# Infer type of true labels
		if isinstance(labels_true, list) :
			len_labels_true = len(labels_true)
			labels_true_type = 'list'
		elif isinstance(labels_true, tuple) :
			len_labels_true = len(labels_true)
			labels_true_type = 'tuple'
		elif isinstance(labels_true, np.ndarray) :
			len_labels_true = labels_true.shape[0]
			labels_true_type = 'np.ndarray'
		elif labels_true is None :
			labels_true_type = 'None'
		else :
			print_screen.PrintStaticInfo('[ERROR] Unsupported input format for true labels.')
			print_screen.PrintStaticInfo('[ERROR] Terminating the program ...')
			sys.exit()

		# Infer type of predicted labels
		if isinstance(labels_pred, list) :
			len_labels_pred = len(labels_pred)
			labels_pred_type = 'list'
		elif isinstance(labels_pred, tuple) :
			len_labels_pred = len(labels_pred)
			labels_pred_type = 'tuple'
		elif isinstance(labels_pred, np.ndarray) :
			len_labels_pred = labels_pred.shape[0]
			labels_pred_type = 'np.ndarray'
		elif labels_pred is None :
			labels_pred_type = 'None'
		else :
			print_screen.PrintStaticInfo('[ERROR] Unsupported input format for predicted labels.')
			print_screen.PrintStaticInfo('[ERROR] Terminating the program ...')
			sys.exit()

		# Check if the labels that have non-None sizes have the sizes consistet with image source inputs
		if labels_true_type != 'None' :
			if len_labels_true != len_image_sources :
				print_screen.PrintStaticInfo('[ERROR] The number of image sources and true labels do NOT match.')		
				print_screen.PrintStaticInfo('[ERROR] Terminating the code ...')
				sys.exit()
		if labels_pred_type != 'None' :
			if len_labels_pred != len_image_sources :
				print_screen.PrintStaticInfo('[ERROR] The number of image sources and predicted labels do NOT match.')		
				print_screen.PrintStaticInfo('[ERROR] Terminating the code ...')
				sys.exit()

		# Infer shapes
		if plot_dim_1 == -1 and plot_dim_2 == -1 :
			dim_1 = int(np.ceil(np.sqrt(len_image_sources)))
			dim_2 = dim_1
		elif plot_dim_1 == -1 :
			dim_2 = int(plot_dim_2)
			dim_1 = int(np.ceil(float(len_image_sources)/dim_2))
		elif plot_dim_2 == -1 :
			dim_1 = int(plot_dim_1)
			dim_2 = int(np.ceil(float(len_image_sources)/dim_1))
		else :
			if int(plot_dim_1)*int(plot_dim_2) != len_image_sources :
				print_screen.PrintStaticInfo('[ERROR] The input shapes do not match.')
				print_screen.PrintStaticInfo('[ERROR] Continuing with default shape of plot ...')
				dim_1 = int(np.ceil(np.sqrt(len_image_sources)))
				dim_2 = dim_1
			else :
				dim_1 = int(plot_dim_1)
				dim_2 = int(plot_dim_2)

		# Create figure and axes
		figure, axes = plt.subplots(dim_1, dim_2) # Create the plot with calculated number of rows and columns

		# Set spaces for height and width
		hspace = 0.4 # Default
		wspace = 0.4 # Default
		# Increment as per need
		if labels_pred is not None :
			hspace += 0.3
		if labels_true is not None :
			hspace += 0.3
		# Adjust the spacing
		figure.subplots_adjust(hspace = hspace, wspace = wspace)

		# If interp is legit, set. Otherwise, error and retract to spline16
		if interp not in ['spline16', 'nearest'] :
			print_screen.PrintStaticInfo('[ERROR] Non-standard method used for interpolation.')
			print_screen.PrintStaticInfo('[ERROR] Continuing with default method of spline16 ...')
			interp = 'spline16'

		# Plot each image one-by-one
		for index, an_axis in enumerate(axes.flat) :

			# There can be more indices. So, plot only till we have data available
			if index < len_image_sources :

				# Fetch image data
				an_image_source = image_sources[index]
				if isinstance(an_image_source, str) :
					# Path is given. Check if it is a .jpg, .png or .npy path
					if an_image_source.strip().endswith('.jpg') or an_image_source.strip().endswith('.jpeg') :
						im = cv2.imread(an_image_source.strip())
						an_axis.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation = interp)
					elif an_image_source.strip().endswith('.png') :
						im = cv2.imread(an_image_source.strip())
						an_axis.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation = interp)
					elif an_image_source.strip().endswith('.npy') :
						im = np.load(an_image_source.strip()).astype(np.uint8)
						shape_im = list(im.shape)
						if len(shape_im) == 3 and shape_im[2] == 3 : # RGB image
							if np.max(im) <= 1.0 :
								if is_rescale_if_out_of_range :
									im = im*255
							# We also need this step that converts the possible floats into np.uint8 format, which is supported by images
							im = im.astype(np.uint8)
							an_axis.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation = interp)
						elif (len(shape_im) == 3 and shape_im[2] == 1) or (len(shape_im) == 2) : # Grayscale
							im = np.squeeze(im)
							if np.max(im) <= 1.0 :
								if is_rescale_if_out_of_range :
									im = im*255
							an_axis.imshow(im, cmap = 'gray')
				elif isinstance(an_image_source, np.ndarray) : # The image is directly fed as np array
					im = an_image_source
					shape_im = list(im.shape)
					if len(shape_im) == 3 and shape_im[2] == 3 : # RGB image
						if np.max(im) <= 1.0 :
							if is_rescale_if_out_of_range :
								im = im*255
						# We also need this step that converts the possible floats into np.uint8 format, which is supported by images
						im = im.astype(np.uint8)
						an_axis.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation = interp)
					elif (len(shape_im) == 3 and shape_im[2] == 1) or (len(shape_im) == 2) : # Grayscale
						im = np.squeeze(im)
						if np.max(im) <= 1.0 :
							if is_rescale_if_out_of_range :
								im = im*255
						an_axis.imshow(im, cmap = 'gray')

				# Fetch label data
				an_x_label = ''
				if labels_true is not None :
					a_label_true = labels_true[index]
					if isinstance(a_label_true, int) or isinstance(a_label_true, np.int32) or isinstance(a_label_true, np.int64) or isinstance(a_label_true, np.int) :
						an_x_label += 'True : ' + str(a_label_true)
					elif isinstance(a_label_true, float) or isinstance(a_label_true, np.float32) or isinstance(a_label_true, np.float64) or isinstance(a_label_true, np.float):
						an_x_label += 'True : ' + str(int(a_label_true))
					elif isinstance(a_label_true, np.ndarray) :
						an_x_label += 'True : ' + str(int(np.argmax(a_label_true)))
					elif isinstance(a_label_true, str) :
						an_x_label += 'True : ' + str(a_label_true.strip())
					else :
						print_screen.PrintStaticInfo('[ERROR] True label information : ' + str(a_label_true) + ' is not understood.')
						an_x_label += 'True : ' + '[ERROR]'
				an_x_label += '\n'
				if labels_pred is not None :
					a_label_pred = labels_pred[index]
					if isinstance(a_label_pred, int) or isinstance(a_label_pred, np.int32) or isinstance(a_label_pred, np.int64) or isinstance(a_label_pred, np.int) :
						an_x_label += 'Pred : ' + str(a_label_pred)
					elif isinstance(a_label_pred, float) or isinstance(a_label_pred, np.float32) or isinstance(a_label_pred, np.float64) or isinstance(a_label_pred, np.float):
						an_x_label += 'Pred : ' + str(int(a_label_pred))
					elif isinstance(a_label_pred, np.ndarray) :
						an_x_label += 'Pred : ' + str(int(np.argmax(a_label_pred)))
					elif isinstance(a_label_pred, str) :
						an_x_label += 'Pred : ' + str(a_label_pred.strip())
					else :
						print_screen.PrintStaticInfo('[ERROR] Pred label information : ' + str(a_label_true) + ' is not understood.')
						an_x_label += 'Pred : ' + '[ERROR]'
				an_axis.set_xlabel(an_x_label)

			# Remove axis ticks
			an_axis.set_xticks([])
			an_axis.set_yticks([])

		# Show the plot!!
		plt.show(figure)

		# # Timer based close # AVOID BECAUSE OF ISSUES
		# if is_timer_close :
		# 	time.sleep(timer_close)
		# 	plt.close(figure)


# Pseudo main
if __name__ == '__main__' :
	from Print_Updating_Info import screenToPrintUpdatingInfo
	print_screen = screenToPrintUpdatingInfo()
	graph_plotter = screenToDisplayCustomPlots()
	im = np.zeros([28, 28])
	for i in range(28) :
		for j in range(28) :
			im[i][j] = int(((float(i)/28) + (float(j)/28))*127.5)
	print_screen.PrintStaticInfo(im)
	im2 = cv2.imread('color.jpg')
	im3 = cv2.imread('color.png')
	im4 = './color.jpg'
	image_sources = [im for i in range(6)]
	image_sources.append(im2)
	image_sources.append(im3)
	image_sources.append(im4)
	image_sources.append(im4)
	image_sources.append(im3)
	image_sources.append(im2)
	# labels_true = [1, 2, np.array([0, 1, 0, 0]), 3.0, 0.0, '0', '1.32', 1.32, -3.4]
	# labels_pred = [1, 2, np.array([0, 1, 0, 0]), 3.0, 0.0, '0', '1.32', 1.32, -3.4]
	labels_true = [1, 2, np.array([0, 1, 0, 0]), 3.0, 0.0, '0', '1.32', 1.32, -3.4, '1.32', 1.32, -3.4]
	labels_pred = [1, 2, np.array([0, 1, 0, 0]), 3.0, 0.0, '0', '1.32', 1.32, -3.4, '1.32', 1.32, -3.4]

	
	graph_plotter.DisplayImagesFromSourcesInCustomPlot(print_screen, image_sources, labels_true, labels_pred)
	graph_plotter.DisplayImagesFromSourcesInCustomPlot(print_screen, image_sources, labels_true, labels_pred, plot_dim_1 = 6, plot_dim_2 = 2, is_close_prev_figs = True)
	graph_plotter.DisplayImagesFromSourcesInCustomPlot(print_screen, image_sources, labels_true, labels_pred, plot_dim_1 = 4, plot_dim_2 = 3, is_close_prev_figs = True)