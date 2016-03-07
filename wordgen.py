#!/usr/bin/python3

from pynn.element import Matrix, Sigmoid, Merger, Fork
from pynn.path import Path
from pynn.network import Network

import math
from random import random, shuffle
import numpy as np

def idx(c):
	a = ord(c)
	if a >= 0x61 and a <= 0x7A:
		return a - 0x61
	if a == 0x27: # '
		return 0x1A
	if a == 0x2D: # -
		return 0x1B
	if a == 0xA:  # CR
		return 0x1C
	return -1

def char(i):
	if i >= 0 and i <= 0x19:
		return chr(i + 0x61)
	if a == 0x1A: # '
		return chr(0x27)
	if a == 0x1B: # -
		return chr(0x2D)
	if a == 0x1C: # CR
		return chr(0xA)
	return ''

net = Network(1, 1)
size = 0x1D
shid = 0x40

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
'''
def err_handler(type, flag):
	raise Exception("Floating point error (%s), with flag %s" % (type, flag))

saved_handler = np.seterrcall(err_handler)
save_err = np.seterr(all='call')
'''
file = open('20k.txt')
words = []
for word in file:
	words.append(word)

del words[0x400:]

for k in range(0x10):
	cost = 0.0
	shuffle(words)

	for j in range(math.floor(len(words)/batch_size)):
		mem = net.Memory()
		mem_stack = [mem]
		exp = net.Experience()

		for i in range(batch_size):
			word = words[j*batch_size + i]
			depth = len(word)

			mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)
			exp.pipes[net._bpath_link[(1, 0)]].data = np.zeros(shid)

			vouts_stack = []

			for l in range(depth - 1):
				a = idx(word[l])
				lin = [0]*size
				lin[a] = 1
				vins = [np.array(lin)]
				
				# feedforward
				(mem, vouts) = net.feedforward(mem, vins)
				mem_stack.append(mem)
				vouts_stack.append(vouts)

			for l in range(depth - 1):
				a = idx(word[depth - l - 1])
				lres = [0]*size
				lres[a] = 1
				vres = np.array(lres)
				vin = vouts_stack.pop()[0]
				vout = np.tanh(vin)
				verrs = [vout - vres]
				cost += np.sum((verrs[0])**2)

				# backpropagate
				net.backprop(exp, mem_stack.pop(), verrs)
		
		#print(net.nodes[1].weight, '\n')
		net.learn(exp, 1e-2/batch_size)

	print(str(k) + ' cost: ' + str(cost/len(words)))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

for j in range(len(alphabet)):

	mem = net.Memory()
	mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)

	a = idx(alphabet[j])
	print(alphabet[j], end='')

	for i in range(0x40):
		lin = [0]*size
		lin[a] = 1
		vins = [np.array(lin)]
		(mem, vouts) = net.feedforward(mem, vins)
		a = np.argmax(vouts[0])
		letter = char(a)

		if letter == '\n':
			break
		print(letter, end='')
	print()
