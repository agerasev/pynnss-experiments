#!/usr/bin/python3

from pynn.element import Matrix, Sigmoid, Merger, Fork
from pynn.path import Path
from pynn.network import Network

import numpy as np

net = Network(1, 1)
sin = 2
shid = 4
sout = 2

net.nodes[0] = Matrix(sin, shid) # W_xh
net.nodes[1] = Matrix(shid, shid) # W_hh
net.nodes[2] = Merger(shid, 2)
net.nodes[3] = Sigmoid(shid)
net.nodes[4] = Fork(shid, 2)
net.nodes[5] = Matrix(shid, sout) # W_hy

net.paths.append(Path((-1, 0), ( 0, 0)))
net.paths.append(Path(( 0, 0), ( 2, 0)))
net.paths.append(Path(( 1, 0), ( 2, 1)))

net.paths.append(Path(( 2, 0), ( 3, 0)))
net.paths.append(Path(( 3, 0), ( 4, 0)))

net.paths.append(Path(( 4, 1), ( 1, 0)))
net.paths.append(Path(( 4, 0), ( 5, 0)))
net.paths.append(Path(( 5, 0), (-1, 0)))

net.update()

mem = net.Memory()
mem.pipes[net._fpath_link[(1, 0)]].data = np.zeros(shid) # put data in value loopback
(mem, _) = net.feedforward(mem, [np.zeros(sin)])

exp = net.Experience()
exp.pipes[net._bpath_link[(1, 0)]].data = np.zeros(shid) # put data in error loopback
net.backprop(exp, mem, [np.zeros(sout)])

# Maybe detect loopbacks and put zero vectors there automatically?
