#!/usr/bin/python3

import math
from copy import copy
from random import random
import numpy as np
import pynn as nn


net = nn.Network(1, 1)
size = 8
shid = 8

net.nodes[0] = nn.MatrixProduct(size, shid) # W_xh
net.nodes[1] = nn.MatrixProduct(shid, shid) # W_hh
net.nodes[2] = nn.Merger(shid, 2)
net.nodes[3] = nn.Bias(shid)
net.nodes[4] = nn.Tanh(shid)
net.nodes[5] = nn.Fork(shid, 2)
net.nodes[6] = nn.MatrixProduct(shid, size) # W_hy

net.paths.append(nn.Path((-1, 0), ( 0, 0)))
net.paths.append(nn.Path(( 0, 0), ( 2, 0)))
net.paths.append(nn.Path(( 1, 0), ( 2, 1)))

net.paths.append(nn.Path(( 2, 0), ( 3, 0)))
net.paths.append(nn.Path(( 3, 0), ( 4, 0)))
net.paths.append(nn.Path(( 4, 0), ( 5, 0)))

net.paths.append(nn.Path(( 5, 1), ( 1, 0)))
net.paths.append(nn.Path(( 5, 0), ( 6, 0)))
net.paths.append(nn.Path(( 6, 0), (-1, 0)))

net.update()

batch_size = 0x10
batches_num = 0x10

init_state = net.newState()
init_state.pipes[net._flink[(1, 0)]].data = np.zeros(shid)

print('learn weights and biases')
for k in range(0x40):
	cost = 0.0

	for j in range(batches_num):
		lcost = 0.0

		grad = net.newGradient()
		#isgrad = np.zeros(shid)
		backprop_count = 0

		for i in range(batch_size):
			a = math.floor(random()*size)
			depth = 16 + math.floor(random()*16)

			state = init_state
			state_stack = [state]

			vouts_stack = []

			for l in range(depth):
				lin = [0]*size
				lin[a] = 1
				vins = [np.array(lin)]
				
				# transmit
				state = copy(state)
				vouts = net.transmit(state, vins)
				state_stack.append(state)
				vouts_stack.append(vouts)

				a += 1
				if a >= size:
					a = 0

			error = net.newError()
			error.pipes[net._blink[(1, 0)]].data = np.zeros(shid)

			for l in range(depth):
				lres = [0]*size
				lres[a] = 1
				vres = np.array(lres)
				verrs = [np.tanh(vouts_stack.pop()[0]) - vres]
				lcost += np.sum((verrs[0])**2)

				# backprop
				net.backprop(grad, error, state_stack.pop(), verrs)
				backprop_count += 1

				a -= 1
				if a < 0:
					a = size - 1

			#isgrad += error.pipes[net._blink[(1, 0)]].data

		lcost /= batch_size
		cost += lcost

		grad.mul(1/backprop_count)
		#isgrad /= batch_size

		clipval = 1e0
		grad.clip(clipval)
		#isgrad = np.clip(isgrad, -clipval, clipval)

		rate = 8e-2 #1e-2*math.sqrt(lcost)
		# print(rate)
		net.learn(grad, rate)
		#init_state.pipes[net._flink[(1, 0)]].data -= 1e-4*isgrad

	print(str(k) + ' cost: ' + str(cost/batches_num))


a = math.floor(random()*size)
depth = 32

state = copy(init_state)

for i in range(depth):
	lin = [0]*size
	lin[a] = 1
	vins = [np.array(lin)]
	vouts = net.transmit(state, vins)
	a = np.argmax(vouts[0])
	print(a, end='')
