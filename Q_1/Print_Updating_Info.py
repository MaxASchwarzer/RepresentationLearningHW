# Dependencies
import os
import sys


# Define a class to hold a printer for updating information
class screenToPrintUpdatingInfo(object) :

	"""
	attributes--

	__init__(self) :
		The constructor
	MakeScreenReadyForMultiLineInfo(self) :
		Function that starts the screen and stores it
	EndScreenForMultiLineInfo(self, is_make_new_screen = False) :
		Function to end screen
	PrintUpdatingMultiLineInfo(self, content, is_continue = True, is_make_new_screen = False) :
		Function to display updating information with multiple lines
	PrintUpdatingSingleLineInfo(self, content, is_continue = True, is_update = True) :
		Function to display updating information with single line
	PrintStaticInfo(self, content) :
		Function to print information
	GetProgressString(self, num, den, n_len_progress = 100, st_complete_char = '=', st_incomplete_char = ' ') :
		Function to get the progress string
	"""


	# Define the constructor
	def __init__(self) :

		"""
		inputs--
		"""

		"""
		outputs--
		"""

		os.system('clear')
		self.is_scr_ready_multi_line = False
		self.is_first_single_line = True


	# Define a function that starts the screen and stores it
	def MakeScreenReadyForMultiLineInfo(self) :

		"""
		inputs--
		"""

		"""
		outputs--
		"""

		# Clear screen
		os.system('clear')

		# Keep the boolean for readiness updated
		self.is_scr_ready_multi_line = True


	# Define a function to end screen
	def EndScreenForMultiLineInfo(self, is_make_new_screen = False) :

		"""
		inputs--

		is_make_new_screen : False
			To create a new screen by clearing all the previous content
		"""

		"""
		outputs--
		"""

		# Keep the boolean for readiness updated
		self.is_scr_ready_multi_line = False

		if is_make_new_screen :
			os.system('clear')


	# Define a function to display updating information with multiple lines
	def PrintUpdatingMultiLineInfo(self, content, is_continue = True, is_make_new_screen = False) :

		"""
		inputs--

		content :
			The multi line content to be printed by overwriting
		is_continue :	
			Whether to continue or not, i.e., if more data is incoming
		is_make_new_screen : False
			Whether to erase all screen data after is_continue turns false. If wanted to be true, RUN THE LOOP ONCE MORE, as the update step eats up the previous output. Manually clean the screen OR else, it will be automatically done in the next step of printing
		"""

		"""
		outputs--
		"""

		# If not ready, make ready
		if not self.is_scr_ready_multi_line :
			self.MakeScreenReadyForMultiLineInfo()

		# Print content
		if self.is_scr_ready_multi_line :
			os.system('clear')
			print(content)


		# If not continue 
		if not is_continue :
			self.EndScreenForMultiLineInfo(is_make_new_screen)


	
	# Define a function to display updating information with single line
	def PrintUpdatingSingleLineInfo(self, content, is_continue = True, is_update = True) :

		"""
		inputs--

		content :
			The single line content to be printed by overwriting. MUST BE SINGLE LINE
		is_continue : True
			Whether to continue or not, i.e., if more data is incoming
		is_update : True
			Whether the content is to be updated. Foolishly put here
		"""

		"""
		outputs--
		"""

		# Remove the lines from the content, if any, and make it single line
		content = str(content.strip().split('\n')[0])

		# If to overwrite, add \r conditionally. We do not want to erase the last line printed, if we want to print updating content
		if is_update :
			if not self.is_first_single_line :
				content = '\r' + content
			else :
				self.is_first_single_line = False

		# Print the content
		sys.stdout.write(content)
		sys.stdout.flush()

		# If we are stopping, we must tell that for next purpose, the incoming line will be first!
		if not is_continue :
			self.is_first_single_line = True
			sys.stdout.write('\n')


	# Define a function to print information
	def PrintStaticInfo(self, content) :

		"""
		inputs--

		content :
			The content to be displayed as is
		"""

		"""
		outputs--
		"""

		print(content)


	# Define a function to get the progress string
	def GetProgressString(self, num, den, n_len_progress = 100, st_complete_char = '=', st_incomplete_char = ' ') :

		"""
		inputs--

		num :
			Numerator of progress
		den :
			Denominator of progress
		n_len_progress : 75
			The progress bar length in integer
		st_complete_char : '='
			The character that shows the progress
		st_incomplete_char : ' '
			The character that shows the remaining
		"""

		"""
		st_content :
			The content string
		"""

		# Get the information string
		st_content = ''

		# Get completion factor
		f_frac = float(num)/den
		f_perc = int(100*f_frac)

		# Count num of '=' and ' '
		n_eq = int(f_frac*n_len_progress)
		n_sp = n_len_progress - n_eq

		st_content += '[INFO]		['
		for i in range(n_eq) :
			st_content += st_complete_char
		for i in range(n_sp) :
			st_content += st_incomplete_char
		st_content += '] Percentage Completed : ' + str(f_perc) + ' %'

		return st_content


# Pseudo main
if __name__ == '__main__' :

	# Dependencies
	import time

	# Create a screen instance
	stdscr = screenToPrintUpdatingInfo()

	# Print updating content
	for i in range(10) :

		if i%2 :
			stdscr.PrintUpdatingMultiLineInfo(str(i) + '\n' + str(i + 1) + '\n' + str(i + 2), i!=9, is_make_new_screen = False) 
			time.sleep(0.5)
		else :
			stdscr.PrintUpdatingMultiLineInfo(str(i) + '\n' + str(i + 1), i!=9, is_make_new_screen = False) 
			time.sleep(0.5)

	# print('1')
	# print('\r2') # DOESN'T WORK AS PER INTENDED

	# Print single line updating info
	for i in range(50) :
		a_str = '['
		for j in range(i) :
			a_str += '#'
		for j in range(50 - 1 - i) :
			a_str += ' '
		a_str += ']'
		stdscr.PrintUpdatingSingleLineInfo(a_str, i != 49)
		time.sleep(0.5)

	print('1')
	print('\r2') # DOESN'T WORK AS PER INTENDED


		