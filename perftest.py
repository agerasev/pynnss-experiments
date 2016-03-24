#!/usr/bin/python3

import pynn as nn
from copy import copy
import numpy as np

data = open('data/witcher_rus.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(data)))

size = len(chars)

ci = {ch: i for i, ch in enumerate(chars)}
ic = {i: ch for i, ch in enumerate(chars)}

seq_len = 25
shid = 100
rate_f = 1e-1

net = nn.Network(1, 1)

net.nodes[0] = nn.MatrixProduct(size, shid)  # W_xh
net.nodes[1] = nn.MatrixProduct(shid, shid)  # W_hh
net.nodes[2] = nn.Merger(shid, 2)
net.nodes[3] = nn.Bias(shid)
net.nodes[4] = nn.Tanh(shid)
net.nodes[5] = nn.Fork(shid, 2)
net.nodes[6] = nn.MatrixProduct(shid, size)  # W_hy
net.nodes[7] = nn.Bias(size)

net.link(nn.Path((-1, 0), (0, 0)))
net.link(nn.Path((0, 0), (2, 0)))
net.link(nn.Path((1, 0), (2, 1)))

net.link(nn.Path((2, 0), (3, 0)))
net.link(nn.Path((3, 0), (4, 0)))
net.link(nn.Path((4, 0), (5, 0)))

net.link(nn.Path((5, 1), (1, 0), np.zeros(shid)))
net.link(nn.Path((5, 0), (6, 0)))
net.link(nn.Path((6, 0), (7, 0)))
net.link(nn.Path((7, 0), (-1, 0)))

rate = nn.RateAdaGrad(net, rate_f)

ff = nn.Node.Profiler()
bp = nn.Node.Profiler()
gd = nn.Node.Profiler()

pos = 0
smooth_loss = -np.log(1/size)*seq_len
epoch = 0

state = net.newState()

loss = 0
grad = net.newGradient()

with ff:
	state_stack = []
	vouts_stack = []

	for i in range(seq_len):
		a = ci[data[pos + i]]
		vin = np.zeros(size)
		vin[a] = 1
		vins = [vin]

		# feedforward
		vouts = net.transmit(state, vins)
		state_stack.append(copy(state))
		vouts_stack.append(vouts)

with bp:
	error = net.newError()

	for i in reversed(range(seq_len)):
		a = ci[data[pos + i + 1]]
		vres = np.zeros(size)
		vres[a] = 1
		vout = vouts_stack.pop()[0]
		evout = np.exp(vout)
		nevout = evout/np.sum(evout)
		verrs = [nevout - vres]
		loss += -np.log(nevout[a])

		# backpropagate
		net.backprop(grad, error, state_stack.pop(), verrs)

with gd:
	grad.clip(5e0)
	rate.update(grad)
	net.learn(grad, rate)

print('ff: %f' % (1e3*ff.time))
print(' ff_net: %f' % (1e3*net.fprof.time))
nt = 0
for _, node in net.nodes.items():
	nt += node.fprof.time
print('  ff_nodes: %f' % (1e3*nt))

print('bp: %f' % (1e3*bp.time))
print(' bp_net: %f' % (1e3*net.bprof.time))
nt = 0
for _, node in net.nodes.items():
	nt += node.bprof.time
print('  bp_nodes: %f' % (1e3*nt))

print('gd: %f' % (1e3*gd.time))
