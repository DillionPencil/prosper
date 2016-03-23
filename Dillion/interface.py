# ClassifierInterface
# the description of each function is the way we want you to implement the function in it's subclass

from abc import ABCMeta, abstractmethod, abstractproperty

class ClassifierInterface(object):	
	__metaclass__=ABCMeta

	@abstractmethod
	def getData(self):
		'''
		input:
			some meta data to desctibe the data you want
		output:
			data
				* for convenience, it should be a dict with humen readable indices
				* the data should be preprocessed in this function
				* all the data you need in the following processes should be returned here
		'''
		pass

	@abstractmethod
	def train(self):
		'''
		input:
			data(the data returned from function 'getData()')
		output:
			model
		'''
		pass

	@abstractmethod
	def test(self):
		'''
		input:
			data(the data returned from function 'getData()')
			model(returned from function 'train()')
		output:
			predictLabels
		'''
		pass

	@abstractmethod
	def evaluate(self):
		'''
		input:
			predictLabels and trueLabels
		output:
			kind of score you define
		'''
		pass