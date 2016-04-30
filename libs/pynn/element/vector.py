#!/usr/bin/python3

import pynn.array as array
from pynn.element import Element


class VectorElement(Element):
	def __init__(self, size, **kwargs):
		Element.__init__(self, size, size, **kwargs)
		self.size = size


class Bias(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def newState(self, factory):
		return self._State(factory.zeros(self.size))

	def _transmit(self, ctx):
		array.add(ctx.dst, ctx.src, ctx.state.data)

	def _backprop(self, ctx):
		if ctx.grad is not None:
			array.radd(ctx.grad.data, ctx.dst)
		array.copy(ctx.src, ctx.dst)


class Uniform(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def _transmit(self, ctx):
		array.copy(ctx.dst, ctx.src)

	def _backprop(self, ctx):
		array.copy(ctx.src, ctx.dst)


class Tanh(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	class _Trace(Element._Trace):
		def __init__(self, odata):
			Element._Trace.__init__(self)
			self.odata = odata

		def set(self, src):
			array.copy(self.odata, src.odata)

	def newTrace(self, factory):
		return self._Trace(factory.empty(self.size))

	def _transmit(self, ctx):
		array.tanh(ctx.dst, ctx.src)
		if ctx.trace is not None:
			array.copy(ctx.trace.odata, ctx.dst)

	def _backprop(self, ctx):
		array.bptanh(ctx.src, ctx.dst, ctx.trace.odata)


class Softmax(VectorElement):
	def __init__(self, size, **kwargs):
		VectorElement.__init__(self, size, **kwargs)

	def _transmit(self, ctx):
		array.softmax(ctx.dst, ctx.src)

	def _backprop(self, ctx):
		raise NotImplementedError()

# TODO:
# class Rectifier(VectorElement)
