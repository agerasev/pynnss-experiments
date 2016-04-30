#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.node import Node


class Element(Node):
	class _State(Node._State):
		def __init__(self, data=None):
			Node._State.__init__(self)
			self.data = data

		def set(self, src):
			array.copy(self.data, src.data)

		class _Gradient(Node._State._Gradient):
			def __init__(self, data):
				Node._State._Gradient.__init__(self)
				self.data = data

			def mul(self, factor):
				array.rmuls(self.data, factor)

			def clip(self, value):
				array.rclip(self.data, -value, value)

			def clear(self):
				array.clear(self.data)

		def newGradient(self, factory):
			if self.data is not None:
				return self._Gradient(factory.zeros(self.data.shape))
			return None

		class _RateConst(Node._State._Rate):
			def __init__(self, factor):
				Node._State._Rate.__init__(self)
				self.factor = factor

			def apply(self, dst, src):
				array.rsubmuls(dst, src, self.factor)

		class _RateAdaGrad(_RateConst):
			def __init__(self, factor, data):
				Element._State._RateConst.__init__(self, factor)
				self.data = data

			def update(self, grad):
				array.radd_adagrad(self.data, grad.data)

			def apply(self, dst, src):
				array.rsub_adagrad(dst, src, self.factor, self.data)

		def newRate(self, factory, rate, adagrad=False):
			if adagrad:
				nparray = np.zeros(self.data.shape) + 1e-8
				return self._RateAdaGrad(rate, factory.copynp(nparray))
			else:
				return self._RateConst(rate)

		def learn(self, grad, rate):
			if grad is not None:
				rate.apply(self.data, grad.data)

	def newState(self, factory):
		return None

	def __init__(self, isizes, osizes, **kwargs):
		Node.__init__(self, isizes, osizes, **kwargs)

	def _transmit(self, ctx):
		raise NotImplementedError()

	def _backprop(self, ctx):
		raise NotImplementedError()
