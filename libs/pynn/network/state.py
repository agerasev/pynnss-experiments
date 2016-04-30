#!/usr/bin/python3

from pynn.network.base import _Nodes, _Paths
from pynn.node import Node


class _State(Node._State, _Nodes, _Paths):
	class _Memory(Node._State._Memory, _Nodes, _Paths):
		def __init__(self, nodes, paths):
			_Nodes.__init__(self, nodes)
			_Paths.__init__(self, paths)
			Node._State._Memory.__init__(self)

		def set(self, src):
			self._setnodes(src)
			self._setpaths(src)

	def newMemory(self, factory):
		nodes = self._fornodes(lambda n: n.newMemory(factory))
		paths = self._forpaths(lambda p: factory.copy(p))
		return self._Memory(nodes, paths)

	class _Error(Node._State._Error, _Nodes, _Paths):
		def __init__(self, nodes, paths):
			_Nodes.__init__(self, nodes)
			_Paths.__init__(self, paths)
			Node._State._Error.__init__(self)

		def set(self, src):
			self._setnodes(src)
			self._setpaths(src)

	def newError(self, factory):
		nodes = self._fornodes(lambda n: n.newError(factory))
		paths = self._forpaths(lambda p: factory.zeros(p.shape))
		return self._Error(nodes, paths)

	# TODO: move to gradient
	class _Gradient(Node._State._Gradient, _Nodes):
		def __init__(self, nodes):
			_Nodes.__init__(self, nodes)
			Node._State._Gradient.__init__(self)

		def mul(self, factor):
			self._fornodes(lambda n: n.mul(factor))

		def clip(self, value):
			self._fornodes(lambda n: n.clip(value))

		def clear(self):
			self._fornodes(lambda n: n.clear())

	def newGradient(self, factory):
		nodes = self._fornodes(lambda n: n.newGradient(factory))
		return self._Gradient(nodes)

	class _Rate(Node._State._Rate, _Nodes):
		def __init__(self, nodes):
			_Nodes.__init__(self, nodes)
			Node._State._Rate.__init__(self)

	class _RateAdaGrad(_Rate):
		def __init__(self, nodes):
			_State._Rate.__init__(self, nodes)

		def update(self, grad):
			for nr, ng in zip(self.nodes, grad.nodes):
				if nr is not None and ng is not None:
					nr.update(ng)

	def newRate(self, factory, *args, **kwargs):
		nodes = self._fornodes(
			lambda n: n.newRate(factory, *args, **kwargs)
			)
		if kwargs.get('adagrad', False):
			return self._RateAdaGrad(nodes)
		else:
			return self._Rate(nodes)

	def __init__(self, nodes, paths):
		_Nodes.__init__(self, nodes)
		_Paths.__init__(self, paths)
		Node._State.__init__(self)

	def learn(self, grad, rate):
		for n, g, r in zip(self.nodes, grad.nodes, rate.nodes):
			if n is not None and g is not None:
				n.learn(g, r)
