#!/usr/bin/python3

import pynn.array as array
from pynn.element import Element


class Mixer(Element):
	class _Context(Element._Context):
		def __init__(self, size, accum, node):
			Element._Context.__init__(self, node)
			self.accum = accum

	def newContext(self, factory):
		return self._Context(self.size, factory.empty(self.size), self)

	def __init__(self, size, inum, onum, **kwargs):
		Element.__init__(self, [size]*inum, [size]*onum, **kwargs)
		self.size = size

	def _transmit(self, ctx):
		array.copy(ctx.accum, ctx.src[0])
		for i in range(1, self.inum):
			array.radd(ctx.accum, ctx.srcs[i])

		for i in range(self.onum):
			array.copy(ctx.dsts[i], ctx.accum)

	def _backprop(self, ctx):
		array.copy(ctx.accum, ctx.dst[0])
		for i in range(1, self.onum):
			array.radd(ctx.accum, ctx.dsts[i])

		for i in range(self.inum):
			array.copy(ctx.srcs[i], ctx.accum)


class Fork(Mixer):
	def newContext(self, factory):
		return Element._Context(self)

	def __init__(self, size, **kwargs):
		Mixer.__init__(self, size, 1, 2, **kwargs)

	def _transmit(self, ctx):
		array.copy(ctx.dsts[0], ctx.src)
		array.copy(ctx.dsts[1], ctx.src)

	def _backprop(self, ctx):
		array.add(ctx.src, ctx.dsts[0], ctx.dsts[1])


class Join(Mixer):
	def newContext(self, factory):
		return Element._Context(self)

	def __init__(self, size, **kwargs):
		Mixer.__init__(self, size, 2, 1, **kwargs)

	def _transmit(self, ctx):
		array.add(ctx.dst, ctx.srcs[0], ctx.srcs[1])

	def _backprop(self, ctx):
		array.copy(ctx.srcs[0], ctx.dst)
		array.copy(ctx.srcs[1], ctx.dst)
