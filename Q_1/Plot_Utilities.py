# Dependencies
import numpy as np
import matplotlib.pyplot as plt


# Define a function to plot functions and lists of numbers
def PlotFunction(x_, y_, c_ = 'red', label_ = 'Function', alpha_ = 0.75) :
	plt.plot(x_, y_, c = c_, label = label_, alpha = alpha_)


# Define a function to plot functions and lists of numbers
def PlotScatter(x_, y_, c_ = 'red', marker_ = 'x', label_ = 'Scatter', alpha_ = 0.75, ms_ = None) :
	if ms_ is None :
		plt.scatter(x_, y_, c = c_, marker = marker_, label = label_, alpha = alpha_)
	else :
		plt.scatter(x_, y_, c = c_, marker = marker_, label = label_, alpha = alpha_, s = ms_)


# Define a function to set the axis equals
def SetPlotAxisEqual() :
	plt.axis('equal')


# Define a function to set the x-axis limits
def SetPlotXLimits(min_val = -10, max_val = 10) :
	plt.xlim(min_val, max_val)


# Define a function to set the x-axis limits
def SetPlotYLimits(min_val = -10, max_val = 10) :
	plt.ylim(min_val, max_val)