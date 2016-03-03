#!/usr/bin/python3

from pynn.element import Product, Sigmoid
from pynn.path import Path
from pynn.network import Network

import math
from random import random
import numpy as np

net = Network(1, 1)

nin = 16
nout = 4

net.nodes[0] = Product(nin, nout)
net.nodes[1] = Sigmoid(nout)

net.paths.append(Path((-1, 0), (0, 0)))
net.paths.append(Path((0, 0), (1, 0)))
net.paths.append(Path((1, 0), (-1, 0)))

net.update()

batch_size = 0x40
batches_num = 0x100

for k in range(0x10):
	cost = 0.0

	for j in range(batches_num):
		mems = [net.Memory()]
		exp = net.Experience()

		for i in range(batch_size):
			a = math.floor(random()*nin)
			lin = [0]*nin
			lin[a] = 1
			vins = [np.array(lin)]
			
			# feedforward
			(mem, vouts) = net.feedforward(mems[len(mems) - 1], vins)
			mems.append(mem)

			lres = [0]*nout
			lres[a%nout] = 1
			vres = np.array(lres)
			verrs = [vouts[0] - vres]
			cost += np.sum((verrs[0])**2)

			# backpropagate
			net.backprop(exp, mems.pop(), verrs)

		net.learn(exp, 1e-2/batch_size)

	print(str(k) + ' cost: ' + str(cost/batch_size/batches_num))