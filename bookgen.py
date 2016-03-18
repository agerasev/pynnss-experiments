#!/usr/bin/python3

from pynn.element import Matrix, Tanh, Merger, Fork
from pynn.path import Path
from pynn.network import Network

import math
from math import floor
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

'''
class RecurrentNetwork(Network):
	def __init__(self, sin, sout, shid):
		Network.__init__(self, 1, 1)

		self.shid = shid

		# nodes
		self.nodes[0] = Matrix(sin, shid)  # W_xh
		self.nodes[1] = Matrix(shid, shid) # W_hh
		self.nodes[2] = Merger(shid, 2)
		self.nodes[3] = Tanh(shid)
		self.nodes[4] = Fork(shid, 2)
		self.nodes[5] = Matrix(shid, sout) # W_hy

		# paths
		self.paths.append(Path((-1, 0), ( 0, 0)))
		self.paths.append(Path(( 0, 0), ( 2, 0)))
		self.paths.append(Path(( 1, 0), ( 2, 1)))
		self.paths.append(Path(( 2, 0), ( 3, 0)))
		self.paths.append(Path(( 3, 0), ( 4, 0)))
		self.paths.append(Path(( 4, 1), ( 1, 0)))
		self.paths.append(Path(( 4, 0), ( 5, 0)))
		self.paths.append(Path(( 5, 0), (-1, 0)))

		self.update()

	def Memory(self):
		mem = Network.Memory(self)
		mem.pipes[2].data = np.zeros(self.shid)
		return mem

	class _Experience(Network._Experience):
		def __init__(self, par):
			self.shid = 0
			Network._Experience.__init__(self, par)

		def _clean(self):
			Network._Experience._clean(self)
			self.pipes[5].data = np.zeros(self.shid)

	def Experience(self, par):
		exp = Network.Experience(self, par)
		exp.shid = self.shid
		exp.pipes[5].data = np.zeros(self.shid)
		return exp
'''

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
elems = []

i = 0
while i < len(book):
	ni = i + floor(32 + 32*random())
	if ni > len(book):
		elems.append(book[i:])
		break
	else:
		elems.append(book[i:ni])
	i = ni

'''
for elem in elems:
	print(elem)
'''

def gen(c, l):
	mem = net.Memory()
	mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid)

	a = ci(c)
	res = c
	for i in range(l):
		lin = [0]*size
		lin[a] = 1
		vins = [np.array(lin)]
		(mem, vouts) = net.feedforward(mem, vins)
		a = np.argmax(vouts[0])
		c = ic(a)
		res += c
	
	return res


batch_size = 10

par = {
	"clip": 0.5,
	"adagrad": True
	}
exp = net.Experience(par)

for k in range(1000):
	cost = 0
	shuffle(elems)

	for j in range(floor(len(elems)/batch_size)):

		for i in range(batch_size):
			mem = net.Memory()
			mem_stack = [mem]

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

			lcost = 0

			for l in range(depth - 1):
				a = ci(elem[depth - l - 1])
				lres = [0]*size
				lres[a] = 1
				vres = np.array(lres)
				vin = vouts_stack.pop()[0]
				vout = np.tanh(vin)
				verrs = [vout - vres]
				lcost += np.sum(verrs[0]**2)/size
				#-np.sum((1 + vres)*np.log(1 + vout) + (1 - vres)*np.log(1 - vout))/size

				# backpropagate
				net.backprop(exp, mem_stack.pop(), verrs)

			cost += lcost/depth
		
		net.learn(exp, 5e-2)
		exp.clean()

	print(str(k) + ' cost: ' + str(100*cost/len(elems)))
	if (k + 1)%10 == 0:
		print(gen('a', 0x80))

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for j in range(len(alphabet)):
	print(gen(alphabet[j], 0x80))
