# Dependencies
import numpy as np
import sys
import gzip
import pickle


# Define a class to hold the data loader for MNIST
class dataLoader() :

	# Constructor
	def __init__(self, batch_size = 256, is_load_from_mnist_pkl_gz = True) :

		"""
		inputs :

		batch_size : 256
			The number of elements in one batch
		is_load_from_mnist_pkl_gz : True
			Whether to load the dataset raw from the mnist.pkl.gz. We do this to avoid pickle issues in python2 and python3 (of which, there are MANY!)
		"""

		# Load the dataset
		if is_load_from_mnist_pkl_gz :
			with gzip.open('mnist.pkl.gz', 'rb') as mnist_gzip :
				mnist_pickled = pickle._Unpickler(mnist_gzip)
				mnist_pickled.encoding = 'latin1'
				train_split, validation_split, test_split = mnist_pickled.load()

			# Create the (randomized training), validation and testing splits (no need to randomize these 2)
			self.x_train = train_split[0]
			self.y_train = train_split[1]
			self.x_validation = validation_split[0]
			self.y_validation = validation_split[1]
			self.x_test = test_split[0]
			self.y_test = test_split[1]
			# # Create a random interation
			# self.iteration_train = np.random.permutation(self.x_train.shape[0])
			# self.x_train = self.x_train[self.iteration_train]
			# self.y_train = self.y_train[self.iteration_train]

			# Create a pointer to current batch start point
			self.current_batch_start_train = 0
			self.current_batch_start_validation = 0
			self.current_batch_start_test = 0

			# Store attribute for batch size
			self.batch_size = batch_size

		# This place should never be reached by any code ...
		else :
			print('[ERROR] Unimplemented option to load directly.')
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


	# Define a method to check if the set has next batch
	def IsNextBatchExists(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check if the next batch exists
		"""

		"""
		outputs :

		is_exists (implicit) :
			Whether there is a next batch in the set
		"""

		# Just check if the current batch start is smaller than the length of split
		if split == 'Train' :
			return self.current_batch_start_train < self.x_train.shape[0]
		elif split == 'Validation' :
			return self.current_batch_start_validation < self.x_validation.shape[0]
		elif split == 'Test' :
			return self.current_batch_start_test < self.x_test.shape[0]
		
		# This place should never be reached by any code ...
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


	# Define a method to get the next batch from the split
	def GetNextBatch(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check if the next batch exists
		"""

		"""
		outputs :

		x_batch :
			The batch of the data. SHAPE : [<batch_size>, <feat_dim>]
		y_batch ;
			The batch of the labels. SHAPE : [<batch_size>, ]
		"""

		# Return the batch and increment the counter
		if split == 'Train' :
			start_point = self.current_batch_start_train
			end_point = np.minimum(start_point + self.batch_size, self.x_train.shape[0])
			x_batch = self.x_train[start_point : end_point]
			y_batch = self.y_train[start_point : end_point]
			self.current_batch_start_train = end_point
		elif split == 'Validation' :
			start_point = self.current_batch_start_validation
			end_point = np.minimum(start_point + self.batch_size, self.x_validation.shape[0])
			x_batch = self.x_validation[start_point : end_point]
			y_batch = self.y_validation[start_point : end_point]
			self.current_batch_start_validation = end_point
		elif split == 'Test' :
			start_point = self.current_batch_start_test
			end_point = np.minimum(start_point + self.batch_size, self.x_test.shape[0])
			x_batch = self.x_test[start_point : end_point]
			y_batch = self.y_test[start_point : end_point]
			self.current_batch_start_test = end_point

		# This place should never be reached by any code ...	
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)

		return x_batch, y_batch


	# Define a method to get the entire data splits
	def GetDataSplit(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check if the next batch exists
		"""

		"""
		outputs :

		x_split (implicit) :
			The split of the data. SHAPE : [<batch_size>, <feat_dim>]
		y_split (implicit) :
			The split of the labels. SHAPE : [<batch_size>, ]
		"""

		if split == 'Train' :
			return self.x_train, self.y_train
		elif split == 'Validation' :
			return self.x_validation, self.y_validation
		elif split == 'Test' :
			return self.x_test, self.y_test
		# This place should never be reached by any code ...	
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


	# Define a function to reset the batch generation
	def ResetDataSplit(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check
		"""

		"""
		outputs :
		"""

		if split == 'Train' :
			self.current_batch_start_train = 0
		elif split == 'Validation' :
			self.current_batch_start_validation = 0
		elif split == 'Test' :
			self.current_batch_start_test = 0
		# This place should never be reached by any code ...	
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


# Define a function to compute stable softmax of a batch of data
def Softmax(feat) :

	"""
	Inputs :

	feat : 
		A feature matrix which needs to be soft-maxed along the columns
	"""

	"""
	Outputs :

	softmax :
		The soft-max of the feature matrix, along the columns
	"""

	# Softmax is translation invariant. Utilize this to evaluate softmax with maximum feature component being 0
	scaled_feats = feat - np.max(feat, axis = 0)
	softmax_num = np.exp(scaled_feats)
	softmax = softmax_num/(1e-10 + np.sum(softmax_num, axis = 0))

	return softmax


# Define a function to compute the sigmoid of a batch of data
def Sigmoid(feat) :

	"""
	Inputs :

	feat : 
		A feature matrix which needs to be soft-maxed along the columns. Shape : [<batch_size>, <feature_dim>]
	"""

	"""
	Outputs :

	sigmoid (implicit) :
		The sigmoid of the feature matrix, taken element-wise. [<batch_size>, <feature_dim>]
	"""

	return 1.0/(1.0 + np.exp(-1.0 * feat))


# Define a function to compute the tanh of a batch of data
def Tanh(feat) :

	"""
	Inputs :

	feat : 
		A feature matrix which needs to be soft-maxed along the columns. Shape : [<batch_size>, <feature_dim>]
	"""

	"""
	Outputs :

	tanh (implicit) :
		The tanh of the feature matrix, taken element-wise. [<batch_size>, <feature_dim>]
	"""

	return np.tanh(feat)


# Define a function to compute the ReLU of a batch of data
def ReLU(feat) :

	"""
	Inputs :

	feat : 
		A feature matrix which needs to be soft-maxed along the columns. Shape : [<batch_size>, <feature_dim>]
	"""

	"""
	Outputs :

	relu (implicit) :
		The relu of the feature matrix, taken element-wise. [<batch_size>, <feature_dim>]
	"""

	return np.maximum(feat, 0.0)


# Define a function to compute the linear map of a batch of data
def ReLU(feat) :

	"""
	Inputs :

	feat : 
		The feature
	"""

	"""
	Outputs :

	linear (implicit) :
		The linear of the input batch
	"""

	return feat


# Define a function to compute the point-wise non-linearity of input, based on the argument
def GetPointwiseGradientOfNonLinearity(feat, non_linearity) :

	"""
	inputs :

	feat :
		The feature for which the non-linearity must be computed
	non_linearity :
		The name of the non-linearity at the current layer. SUPPORT : 'Sigmoid', 'Tanh', 'ReLU', 'Linear'
	"""

	if non_linearity == 'ReLU' :
		return np.array(feat > 0).astype(np.float32)
	elif non_linearity == 'Sigmoid' :
		return feat * (1.0 - feat)
	elif non_linearity == 'Tanh' :
		return 1.0 - np.power(feat, 2)
	elif non_linearity == 'Linear' : # Solely for input!!
		return feat
	else :
		# The control should never reach here
		print('[ERROR] Unimplemented Non-Linearity Encountered at Current Layer :', non_linearity)
		print('[ERROR] Terminating the code ...')
		sys.exit()


# Pseudo-main
if __name__ == '__main__' :
	
	print('##################################################')
	print('########## TEST : Numerically Stable Softmax')
	print('##################################################')
	m = np.array([[1e100, 1e100, 1e100], [1e10, 1e-9, 1e100]])
	print('m = \n', m)
	print('Softmax(m) = \n', Softmax(m))

	print('##################################################')
	print('########## TEST : Sigmoid')
	print('##################################################')
	m = np.random.normal(10, 100, [784, 256])
	sigm_m = Sigmoid(m)
	print(sigm_m.shape)
	print(np.all(sigm_m <= 1.0) and np.all(sigm_m >= 0.0))
	print(np.max(sigm_m), np.min(sigm_m))

	print('##################################################')
	print('########## TEST : Tanh')
	print('##################################################')
	m = np.random.normal(10, 100, [784, 256])
	tanh_m = Tanh(m)
	print(tanh_m.shape)
	print(np.all(tanh_m <= 1.0) and np.all(tanh_m >= 0.0))
	print(np.all(tanh_m <= 1.0) and np.all(tanh_m >= -1.0))
	print(np.max(tanh_m), np.min(tanh_m))

	print('##################################################')
	print('########## TEST : ReLU')
	print('##################################################')
	m = np.random.normal(0, 100, [784, 256])
	tanh_m = ReLU(m)
	print(tanh_m.shape)
	print(np.all(tanh_m >= 0.0))
	print(np.max(tanh_m), np.min(tanh_m))	

	print('##################################################')
	print('########## TEST : Data Loader')
	print('##################################################')
	mnist = dataLoader(batch_size = 500)
	print(mnist.x_train.shape)
	print(mnist.y_train.shape)
	print(mnist.x_validation.shape)
	print(mnist.y_validation.shape)
	print(mnist.x_test.shape)
	print(mnist.y_test.shape)
	i = 0
	while mnist.IsNextBatchExists() :
		i += 1
		x_tr, y_tr = mnist.GetNextBatch()
		print('[DEBUG] Training Batch :\t' + str(i) + '\t\tBatch Shape : ', x_tr.shape, y_tr.shape)
	train_data = mnist.GetDataSplit('Train')
	print(train_data[0].shape, train_data[1].shape)
	validation_data = mnist.GetDataSplit('Validation')
	print(validation_data[0].shape, validation_data[1].shape)
	test_data = mnist.GetDataSplit('Test')
	print(test_data[0].shape, test_data[1].shape)
	print('[DEBUG] Batch Max Value : ', np.max(mnist.x_train), ' Batch Min Value : ', np.min(mnist.x_test))
