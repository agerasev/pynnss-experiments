#!/usr/bin/python3

import pynn as nn

import math
from random import random
import numpy as np

net = nn.Network(1, 1)

nin = 4
nout = 2

net.nodes[0] = nn.MatrixProduct(nin, nout)

net.paths.append(nn.Path((-1, 0), ( 0, 0)))
net.paths.append(nn.Path(( 0, 0), (-1, 0)))

net.update()

batch_size = 0x10
batches_num = 0x100

for k in range(0x10):
	cost = 0.0

	for j in range(batches_num):
		grad = net.newGradient()

		for i in range(batch_size):
			state = net.newState()
			error = net.newError()

			a = math.floor(random()*nin)
			lin = [0]*nin
			lin[a] = 1
			vins = [np.array(lin)]
			
			# feedforward
			vouts = net.transmit(state, vins)

			lres = [0]*nout
			lres[a%nout] = 1
			vres = np.array(lres)
			verrs = [vouts[0] - vres]
			cost += np.sum((verrs[0])**2)

			# backpropagate
			net.backprop(grad, error, state, verrs)

		grad.mul(1/batch_size)
		net.learn(grad, 1e-2)

	print(str(k) + ' cost: ' + str(cost/batch_size/batches_num))

print(net.nodes[0].weight)

test = False
while test:
	num = int(input('Enter number from 0 to ' + str(nin - 1) + ' (-1 to exit): '))
	if num < 0:
		break
	lin = [0]*nin
	lin[num] = 1
	vins = [np.array(lin)]
	(mem, vouts) = net.feedforward(net.Memory(), vins)
	print('Result: ' + str(np.argmax(vouts[0])))