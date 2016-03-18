#!/usr/bin/python3

import math
from random import random
import numpy as np
import pynn as nn

node = nn.MatrixProduct(1, 1)
node.weight[0] = -2

for i in range(0x10):
	for j in range(0x20):
		state = node.newState()
		vins = [np.array([1])]
		
		# feedforward
		vouts = node.transmit(state, vins)

		vres = np.array([1])
		verrs = [(np.tanh(vouts[0]) - vres)/(np.cosh(vouts[0])**2)]
		cost = np.sum((verrs[0])**2)

		# backpropagate
		error = node.newError()
		grad = node.newGradient()
		node.backprop(grad, error, state, verrs)
		node.learn(grad, 1e-2)

	print('#'*math.floor(0x20*(2 + node.weight[0][0])))