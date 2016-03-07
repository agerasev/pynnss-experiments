#!/usr/bin/python3

from pynn.element import Matrix, Sigmoid, Merger, Fork
from pynn.path import Path
from pynn.network import Network

import math
from random import random
import numpy as np

net = Network(1, 1)
size = 8
shid = 8

net.nodes[0] = Matrix(size, shid) # W_xh
net.nodes[1] = Matrix(shid, shid) # W_hh
net.nodes[2] = Merger(shid, 2)
net.nodes[3] = Sigmoid(shid)
net.nodes[4] = Fork(shid, 2)
net.nodes[5] = Matrix(shid, size) # W_hy

net.paths.append(Path((-1, 0), ( 0, 0)))
net.paths.append(Path(( 0, 0), ( 2, 0)))
net.paths.append(Path(( 1, 0), ( 2, 1)))

net.paths.append(Path(( 2, 0), ( 3, 0)))
net.paths.append(Path(( 3, 0), ( 4, 0)))

net.paths.append(Path(( 4, 1), ( 1, 0)))
net.paths.append(Path(( 4, 0), ( 5, 0)))
net.paths.append(Path(( 5, 0), (-1, 0)))

net.update()

batch_size = 0x10
batches_num = 0x10

for k in range(0x10):
	cost = 0.0

	for j in range(batches_num):
		mem = net.Memory()
		mem_stack = [mem]
		exp = net.Experience()

		for i in range(batch_size):
			a = math.floor(random()*size)
			depth = 16 + math.floor(random()*16)

			mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)
			exp.pipes[net._bpath_link[(1, 0)]].data = np.zeros(shid)

			vouts_stack = []

			for l in range(depth):
				lin = [0]*size
				lin[a] = 1
				vins = [np.array(lin)]
				
				# feedforward
				(mem, vouts) = net.feedforward(mem, vins)
				mem_stack.append(mem)
				vouts_stack.append(vouts)

				a += 1
				if a >= size:
					a = 0

			for l in range(depth):
				lres = [0]*size
				lres[a] = 1
				vres = np.array(lres)
				verrs = [vouts_stack.pop()[0] - vres]
				cost += np.sum((verrs[0])**2)

				# backpropagate
				net.backprop(exp, mem_stack.pop(), verrs)

				a -= 1
				if a < 0:
					a = size - 1

		net.learn(exp, 1e-2/batch_size)

	print(str(k) + ' cost: ' + str(cost/batch_size/batches_num))


a = math.floor(random()*size)
depth = 32

mem = net.Memory()
mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)

for i in range(depth):
	lin = [0]*size
	lin[a] = 1
	vins = [np.array(lin)]
	(mem, vouts) = net.feedforward(mem, vins)
	a = np.argmax(vouts[0])
	print(a, end='')
