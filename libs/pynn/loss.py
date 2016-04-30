#!/usr/bin/python3

from pynn import array
from pynn.node import Node
from pynn.element.vector import Softmax


class Loss:
	class _Context:
		def __init__(self):
			self.loss = 0.

	def __init__(self):
		pass

	def nodetype(self):
		return Node


class SoftmaxLoss(Loss, Softmax):
	class _Context(Loss._Context, Softmax._Context):
		def __init__(self, node):
			Loss._Context.__init__(self)
			Softmax._Context.__init__(self, node)

	def newContext(self, factory):
		return self._Context(self)

	class _Trace(Softmax._Trace):
		def __init__(self, odata):
			Softmax._Trace.__init__(self)
			self.odata = odata

		def set(self, src):
			array.copy(self.odata, src.odata)

	def newTrace(self, factory):
		return self._Trace(factory.empty(self.size))

	def __init__(self, size, **kwargs):
		Loss.__init__(self)
		Softmax.__init__(self, size, **kwargs)

	def _transmit(self, ctx):
		if ctx.trace is not None:
			array.softmax(ctx.trace.odata, ctx.src)
			if ctx.dst is not None:
				array.copy(ctx.dst, ctx.trace.odata)
		else:
			Softmax._transmit(self, ctx)

	def _backprop(self, ctx):
		ctx.loss += array.softmaxloss(ctx.src, ctx.trace.odata, ctx.dst)

	def nodetype(self):
		return Softmax
