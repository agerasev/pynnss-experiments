#!/usr/bin/python3

from pynn import array
from pynn.node import Node
from pynn.loss import Loss
from pynn.network.base import _Nodes, _Paths


class _Context(Node._Context, Loss._Context, _Nodes, _Paths):
	class _Srcs:
		def __init__(self, outer):
			self.outer = outer

		def __getitem__(self, key):
			return self.outer._srcs[key]

		def __setitem__(self, key, value):
			self.outer._srcs[key] = value
			did = self.outer.node.ipaths[key].dst
			self.outer.nodes[did[0]].srcs[did[1]] = value

	class _Dsts:
		def __init__(self, outer):
			self.outer = outer

		def __getitem__(self, key):
			return self.outer._dsts[key]

		def __setitem__(self, key, value):
			self.outer._dsts[key] = value
			sid = self.outer.node.opaths[key].src
			self.outer.nodes[sid[0]].dsts[sid[1]] = value

	def _ssrcs(self, srcs):
		for i in range(self.node.inum):
			self.srcs[i] = srcs[i]

	def _gsrcs(self):
		return self._Srcs(self)

	srcs = property(_gsrcs, _ssrcs)

	def _sdsts(self, dsts):
		for i in range(self.node.onum):
			self.dsts[i] = dsts[i]

	def _gdsts(self):
		return self._Dsts(self)

	dsts = property(_gdsts, _sdsts)

	def _gattr(self, name):
		return getattr(self, '_' + name)

	def _sattr(self, name, value):
		if value is None:
			nvs = [None]*len(self.nodes)
		else:
			nvs = value.nodes
		for nc, nv in zip(self.nodes, nvs):
			if nc is not None:
				setattr(nc, name, nv)
		setattr(self, '_' + name, value)

	state = property(
		lambda self: self._gattr('state'),
		lambda self, value: self._sattr('state', value)
		)

	trace = property(
		lambda self: self._gattr('trace'),
		lambda self, value: self._sattr('trace', value)
		)

	grad = property(
		lambda self: self._gattr('grad'),
		lambda self, value: self._sattr('grad', value)
		)

	rate = property(
		lambda self: self._gattr('rate'),
		lambda self, value: self._sattr('rate', value)
		)

	def __init__(self, node, nodes, paths):
		_Nodes.__init__(self, nodes)
		_Paths.__init__(self, paths)
		self._srcs = [None]*node.inum
		self._dsts = [None]*node.onum
		Node._Context.__init__(self, node)
		Loss._Context.__init__(self)
		for path, arr in zip(self.node.paths, self.paths):
			src = path.src
			dst = path.dst
			self.nodes[src[0]].dsts[src[1]] = arr
			self.nodes[dst[0]].srcs[dst[1]] = arr

	def setmem(self, mem):
		for nc, nm in zip(self.nodes, mem.nodes):
			if nc is not None:
				nc.setmem(nm)
		for pc, pm in zip(self.paths, mem.paths):
			if pm is not None:
				array.copy(pc, pm)

	def getmem(self, mem):
		for nc, nm in zip(self.nodes, mem.nodes):
			if nc is not None:
				nc.getmem(nm)
		for pc, pm in zip(self.paths, mem.paths):
			if pm is not None:
				array.copy(pm, pc)

	def seterr(self, err):
		for nc, ne in zip(self.nodes, err.nodes):
			if nc is not None:
				nc.seterr(ne)
		for pc, pe in zip(self.paths, err.paths):
			if pe is not None:
				array.copy(pc, pe)

	def geterr(self, err):
		for nc, ne in zip(self.nodes, err.nodes):
			if nc is not None:
				nc.geterr(ne)
		for pc, pe in zip(self.paths, err.paths):
			if pe is not None:
				array.copy(pe, pc)

	def _gloss(self):
		loss = 0.
		for node in self.nodes:
			if node is not None and hasattr(node, 'loss'):
				loss += node.loss
		return loss

	def _sloss(self, loss):
		for node in self.nodes:
			if node is not None and hasattr(node, 'loss'):
				node.loss = loss

	loss = property(_gloss, _sloss)
