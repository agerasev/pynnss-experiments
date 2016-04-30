#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.element import Element


class MatrixElement(Element):
	def __init__(self, isize, osize, **kwargs):
		Element.__init__(self, isize, osize, **kwargs)


class Matrix(MatrixElement):
	def __init__(self, isize, osize, **kwargs):
		MatrixElement.__init__(self, isize, osize, **kwargs)

	class _State(Element._State):
		def __init__(self, data):
			Element._State.__init__(self, data)

	def newState(self, factory):
		rand = 0.01*np.random.randn(self.isize, self.osize)
		return self._State(factory.copynp(rand))

	class _Trace(Element._Trace):
		def __init__(self, idata):
			Element._Trace.__init__(self)
			self.idata = idata

		def set(self, src):
			array.copy(self.idata, src.idata)

	def newTrace(self, factory):
		return self._Trace(factory.empty(self.isize))

	def _transmit(self, ctx):
		if ctx.trace is not None:
			array.copy(ctx.trace.idata, ctx.src)
		array.dot(ctx.dst, ctx.src, ctx.state.data)

	def _backprop(self, ctx):
		if ctx.grad is not None:
			array.raddouter(ctx.grad.data, ctx.trace.idata, ctx.dst)
		array.dot(ctx.src, ctx.state.data, ctx.dst)
