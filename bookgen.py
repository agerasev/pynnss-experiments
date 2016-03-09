#!/usr/bin/python3

from pynn.element import Matrix, Tanh, Merger, Fork
from pynn.path import Path
from pynn.network import Network

import math
from random import random, shuffle
import numpy as np

def ci(c):
	if c == '\n':
		return 0x5F
	return ord(c) - 0x20

def ic(i):
	if i == 0x5F:
		return '\n'
	return chr(i + 0x20)

net = Network(1, 1)
size = 0x60
shid = 0x100

net.nodes[0] = Matrix(size, shid) # W_xh
net.nodes[1] = Matrix(shid, shid) # W_hh
net.nodes[2] = Merger(shid, 2)
net.nodes[3] = Tanh(shid)
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

'''
def err_handler(type, flag):
	raise Exception("Floating point error (%s), with flag %s" % (type, flag))

saved_handler = np.seterrcall(err_handler)
save_err = np.seterr(all='call')
'''

book = open('harry_potter_and_the_sorcerers_stone.txt').read(0x1000)
parts = []
sents = []
elems = []

i = 0
while i >= 0:
	ni = book.find('\n', i)
	if ni < 0:
		parts.append(book[i:])
		break
	parts.append(book[i:ni])
	i = ni + 1

for part in parts:
	if len(part) < 1:
		continue
	i = 0
	while i >= 0:
		ecs = '.?!'
		ni = -1
		for c in ecs:
			ti = part.find(c, i)
			if ti >= 0 and (ni < 0 or ti < ni):
				ni = ti
		if ni < 0:
			sents.append(part[i:])
			break
		sents.append(part[i:(ni+1)])
		i = ni + 2

for elem in sents:
	if len(elem) > 1:
		elems.append(elem)

'''
for elem in elems:
	print(elem)
'''

batch_size = 0x1

for k in range(0x10):
	cost = 0.0
	shuffle(elems)

	for j in range(math.floor(len(elems)/batch_size)):
		mem = net.Memory()
		mem_stack = [mem]
		exp = net.Experience()

		for i in range(batch_size):
			elem = elems[j*batch_size + i]
			depth = len(elem)

			mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)
			exp.pipes[net._bpath_link[(1, 0)]].data = np.zeros(shid)

			vouts_stack = []

			for l in range(depth - 1):
				a = ci(elem[l])
				lin = [0]*size
				#print((ic(a)))
				lin[a] = 1
				vins = [np.array(lin)]
				
				# feedforward
				(mem, vouts) = net.feedforward(mem, vins)
				mem_stack.append(mem)
				vouts_stack.append(vouts)

			for l in range(depth - 1):
				a = ci(elem[depth - l - 1])
				lres = [0]*size
				lres[a] = 1
				vres = np.array(lres)
				vin = vouts_stack.pop()[0]
				vout = np.tanh(vin)
				verrs = [vout - vres]
				cost += np.sum((verrs[0])**2)

				# backpropagate
				net.backprop(exp, mem_stack.pop(), verrs)
		
		exp.clip(5e0)
		net.learn(exp, 1e-3/batch_size)

	print(str(k) + ' cost: ' + str(cost/len(elems)))

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for j in range(len(alphabet)):

	mem = net.Memory()
	mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)

	a = ci(alphabet[j])
	print(alphabet[j], end='')

	for i in range(0x80):
		lin = [0]*size
		lin[a] = 1
		vins = [np.array(lin)]
		(mem, vouts) = net.feedforward(mem, vins)
		a = np.argmax(vouts[0])
		c = ic(a)
		print(c, end='')
	print()
