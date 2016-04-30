#!/usr/bin/python3

from pynn.node import Node
from pynn.network.base import _Nodes, _Paths
from pynn.network.state import _State
from pynn.network.context import _Context


class Path:
	def __init__(self, src, dst, size=-1, mem=False):
		self.src = src
		self.dst = dst
		self.mem = mem


class Network(Node, _Nodes, _Paths):
	class _Path:
		def __init__(self, src, dst, size, mem):
			self.src = src
			self.dst = dst
			self.size = size
			self.mem = mem

	class _State(_State):
		def __init__(self, nodes, paths):
			_State.__init__(self, nodes, paths)

	def newState(self, factory):
		nodes = [n.newState(factory) for n in self.nodes]
		paths = []
		for path in self.paths:
			data = None
			if path.mem:
				data = factory.zeros(path.size)
			paths.append(data)
		return self._State(nodes, paths)

	class _Trace(Node._Trace, _Nodes):
		def __init__(self, nodes):
			_Nodes.__init__(self, nodes)
			Node._Trace.__init__(self)

		def set(self, src):
			self._setnodes(src)

	def newTrace(self, factory):
		return self._Trace([n.newTrace(factory) for n in self.nodes])

	class _Context(_Context):
		def __init__(self, node, nodes, paths):
			_Context.__init__(self, node, nodes, paths)

	def newContext(self, factory):
		nodes = [n.newContext(factory) for n in self.nodes]
		paths = [factory.empty(p.size) for p in self.paths]
		return self._Context(self, nodes, paths)

	def __init__(self, isizes, osizes, **kwargs):
		_Nodes.__init__(self)
		_Paths.__init__(self)
		Node.__init__(self, isizes, osizes, **kwargs)
		self.ipaths = [None]*self.inum
		self.opaths = [None]*self.onum
		self._flink = {}
		self._blink = {}
		self.order = None

	def addnodes(self, nodes):
		if isinstance(nodes, Node):
			nodes = [nodes]
		for node in nodes:
			self.nodes.append(node)

	def _nodeid(self, nid):
		pos = 0
		if type(nid) == tuple:
			key = nid[0]
			pos = nid[1]
		else:
			key = nid
		if key < 0 or key >= len(self.nodes):
			raise Exception('no node with key %d' % key)
		node = self.nodes[key]
		return (key, pos), node

	def _snodeid(self, sid):
		(key, pos), node = self._nodeid(sid)
		if pos < 0 or pos >= node.onum:
			raise Exception('wrong opos %d for node %d' % (pos, key))
		return (key, pos), node, node.osizes[pos]

	def _dnodeid(self, did):
		(key, pos), node = self._nodeid(did)
		if pos < 0 or pos >= node.inum:
			raise Exception('wrong ipos %d for node %d' % (pos, key))
		return (key, pos), node, node.isizes[pos]

	def _sisfree(self, src):
		if src in self._flink.keys():
			raise Exception('output (%d,%d) already connected' % src)

	def _disfree(self, dst):
		if dst in self._blink.keys():
			raise Exception('input (%d,%d) already connected' % dst)

	def connect(self, paths):
		if isinstance(paths, Path):
			paths = [paths]
		for path in paths:
			src, snode, ssize = self._snodeid(path.src)
			dst, dnode, dsize = self._dnodeid(path.dst)
			if ssize != dsize:
				raise Exception('sizes dont match')
			self._sisfree(src)
			self._disfree(dst)
			self.paths.append(self._Path(src, dst, ssize, path.mem))
			self._flink[src] = len(self.paths) - 1
			self._blink[dst] = len(self.paths) - 1

	def setinputs(self, dids):
		if type(dids) is int or type(dids) is tuple:
			dids = [dids]
		for i, did in enumerate(dids):
			dst, dnode, dsize = self._dnodeid(did)
			self._disfree(dst)
			if dsize != self.isizes[i]:
				raise Exception('sizes dont match')
			self.ipaths[i] = Path(None, dst, dsize)
			self._blink[dst] = -1

	def setoutputs(self, sids):
		if type(sids) is int or type(sids) is tuple:
			sids = [sids]
		for i, sid in enumerate(sids):
			src, snode, ssize = self._snodeid(sid)
			self._sisfree(src)
			if ssize != self.osizes[i]:
				raise Exception('sizes dont match')
			self.opaths[i] = Path(src, None, ssize)
			self._flink[src] = -1

	class _NodeInfo:
		def __init__(self, inum, onum):
			self.flag = False
			self.iflags = [False]*inum
			self.oflags = [False]*onum

		def transmit(self):
			if not self.flag:
				if all(self.iflags):
					self.flag = True
					self.oflags = [True for _ in self.oflags]
					return True
			return False

	def prepare(self):
		for node in self.nodes:
			if hasattr(node, 'prepare'):
				node.prepare()

		nodes = [self._NodeInfo(n.inum, n.onum) for n in self.nodes]

		for ip in self.ipaths:
			nodes[ip.dst[0]].iflags[ip.dst[1]] = True
		for p in self.paths:
			if p.mem:
				nodes[p.dst[0]].iflags[p.dst[1]] = True

		order = []
		check = 1
		while check > 0:
			check = 0
			for i, n in enumerate(nodes):
				if n.transmit():
					for j in range(len(n.oflags)):
						npi = self._flink[(i, j)]
						if npi >= 0:
							dst = self.paths[npi].dst
							nodes[dst[0]].iflags[dst[1]] = True
					check += 1
					order.append(i)

		activated = [n.flag for n in nodes]
		if not all(activated):
			nnl = filter(lambda n: not n[1], enumerate(activated))
			nns = str([n[0] for n in nnl])
			raise Exception('nodes ' + nns + ' were not activated')

		outputs = [nodes[op.src[0]].oflags[op.src[1]] for op in self.opaths]
		if not all(outputs):
			onl = filter(lambda o: not o[1], enumerate(outputs))
			ons = str([o[0] for o in onl])
			raise Exception('outputs ' + ons + ' are not ready')

		self.order = order

	def _transmit(self, ctx):
		znc = list(zip(self.nodes, ctx.nodes))
		for i in self.order:
			n, nc = znc[i]
			n.transmit(nc)

	def _backprop(self, ctx):
		znc = list(zip(self.nodes, ctx.nodes))
		for i in reversed(self.order):
			n, nc = znc[i]
			n.backprop(nc)
