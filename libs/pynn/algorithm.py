#!/usr/bin/python3


class Feeder:
	def __init__(self, factory, net, state, imem=None, **kwargs):
		self.factory = factory

		self.net = net
		net.prepare()

		self.state = state

		self.ctx = net.newContext(factory)
		self.ctx.state = state

		self.ctx.src = factory.empty(net.isize)
		self.ctx.dst = factory.empty(net.osize)

		self.imem = state.newMemory(factory)

	src = property(lambda self: self.ctx.src)
	dst = property(lambda self: self.ctx.dst)

	def feed(self):
		self.ctx.setmem(self.imem)
		while True:
			self.net.transmit(self.ctx)
			yield


class Teacher:
	def __init__(self, factory, data, net, state=None, **kwagrs):
		self.factory = factory

		self.data = data

		self.net = net
		net.prepare()

		if state is None:
			state = net.newState(factory)
		self.state = state

		self.ctx = net.newContext(factory)
		self.ctx.state = state
		self.ctx.trace = net.newTrace(factory)
		self.ctx.grad = state.newGradient(factory)
		self.ctx.rate = state.newRate(
			factory, kwagrs.get('rate', 1e-1),
			adagrad=kwagrs.get('adagrad', True)
		)
		self.clip = kwagrs.get('clip', 5e0)

		self.traces = None
		maxlen = kwagrs.get('maxlen', 1)
		if maxlen > 1:
			self.traces = [self.net.newTrace(self.factory) for _ in range(maxlen)]

		self.ctx.src = factory.empty(net.isize)
		self.ctx.dst = factory.empty(net.osize)

		self.imem = state.newMemory(factory)
		self.mem = state.newMemory(factory)
		self.mem.set(self.imem)

		self.ierr = state.newError(factory)

		self.teachgen = self._TeachGen()

		self.bmon = kwagrs.get('bmon', None)
		self.emon = kwagrs.get('emon', None)

		self.smem = kwagrs.get('smem', False)

	def _batch(self, batch):
		ctx = self.ctx
		ctx.grad.clear()
		ctx.loss = 0.

		for series in batch:
			ctx.setmem(self.mem)
			for i, entry in enumerate(series):
				entry.getinput(ctx.src)
				self.net.transmit(ctx)
				self.traces[i].set(ctx.trace)
			if self.smem:
				ctx.getmem(self.mem)

			ctx.seterr(self.ierr)
			for i, entry in reversed(list(enumerate(series))):
				entry.getouptut(ctx.dst)
				ctx.trace.set(self.traces[i])
				self.net.backprop(ctx)

		ctx.grad.mul(1/len(batch))
		ctx.grad.clip(self.clip)
		if hasattr(ctx.rate, 'update'):
			ctx.rate.update(ctx.grad)
		ctx.state.learn(ctx.grad, ctx.rate)

	def _EpochGen(self, epoch):
		if self.smem:
			self.mem.set(self.imem)
		for batch in epoch:
			self._batch(batch)
			try:
				if self.bmon is not None:
					self.bmon(self)
			except StopIteration:
				yield

	def _TeachGen(self):
		for epoch in self.data:
			epochgen = self._EpochGen(epoch)
			try:
				while True:
					next(epochgen)
					yield
			except StopIteration:
				try:
					if self.emon is not None:
						self.emon(self)
				except StopIteration:
					yield

	def teach(self):
		next(self.teachgen)
		return self.ctx
