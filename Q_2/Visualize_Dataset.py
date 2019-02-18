# Dependencies
import numpy as np
import Display_Custom_Plots as plot_utils
import Print_Updating_Info as print_utils
import Utilities as utils


# Pseudo-main
if __name__ == '__main__' :

	data_loader = utils.dataLoader(batch_size = 16)

	x_tr, y_tr = data_loader.GetNextBatch('Train')
	x_tr = np.reshape(x_tr, [-1, 28, 28])
	x_val, y_val = data_loader.GetNextBatch('Validation')
	x_val = np.reshape(x_val, [-1, 28, 28])
	x_te, y_te = data_loader.GetNextBatch('Test')
	x_te = np.reshape(x_te, [-1, 28, 28])

	print_screen = print_utils.screenToPrintUpdatingInfo()
	plot_screen = plot_utils.screenToDisplayCustomPlots()

	plot_screen.DisplayImagesFromSourcesInCustomPlot(print_screen = print_screen, image_sources = x_tr, labels_true = y_tr)
	plot_screen.DisplayImagesFromSourcesInCustomPlot(print_screen = print_screen, image_sources = x_val, labels_true = y_val)
	plot_screen.DisplayImagesFromSourcesInCustomPlot(print_screen = print_screen, image_sources = x_te, labels_true = y_te)
