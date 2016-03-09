#!/usr/bin/python3

from pynn.element import Matrix
from pynn.path import Path
from pynn.network import Network

import math
from random import random
import numpy as np

node = Matrix(1, 1)
node.weight[0] = -2

for i in range(0x10):
	for j in range(0x20):
		vins = [np.array([1])]
		
		# feedforward
		(mem, vouts) = node.feedforward(node.Memory(), vins)


		vres = np.array([1])
		verrs = [(np.tanh(vouts[0]) - vres)/(np.cosh(vouts[0])**2)]
		cost = np.sum((verrs[0])**2)

		# backpropagate
		exp = node.Experience()
		node.backprop(exp, mem, verrs)
		node.learn(exp, 1e-2)

	print('#'*math.floor(0x20*(2 + node.weight[0][0])))