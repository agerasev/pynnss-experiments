#!/usr/bin/python3

import numpy as np
from pynn.profile import Profiler


class _Array:
	def __init__(self, shape, dtype):
		self.shape = shape
		self.dtype = dtype

	def get(self):
		raise NotImplementedError()

	def set(self, data):
		raise NotImplementedError()


class _ArrayCPU(_Array):
	def __init__(self, nparray):
		_Array.__init__(self, nparray.shape, nparray.dtype)
		self.np = nparray

	def get(self):
		return np.copy(self.np)

	def set(self, data):
		np.copyto(self.np, data)


class _Factory:
	def __init__(self, dtype=np.float64):
		self.dtype = dtype


class _FactoryCPU(_Factory):
	def __init__(self, dtype=np.float64):
		_Factory.__init__(self, dtype)

	def empty(self, shape):
		return _ArrayCPU(np.empty(shape, dtype=self.dtype))

	def zeros(self, shape):
		return _ArrayCPU(np.zeros(shape, dtype=self.dtype))

	def copy(self, array):
		return _ArrayCPU(np.array(array.np, dtype=self.dtype))

	def copynp(self, nparray):
		return _ArrayCPU(np.array(nparray, dtype=self.dtype))


def newFactory(dtype=None, gpu=False):
	return _FactoryCPU(dtype=(np.float64 if dtype is None else dtype))


names = [
	'clear',
	'copy',
	'add',
	'radd',
	'clip',
	'rclip',
	'muls',
	'mul',
	'rmuls',
	'rmul',
	'dot',
	'raddouter',
	'rsubmuls',
	'rsubmul',
	'tanh',
	'bptanh',
	'softmax',
	'softmaxloss',
	'radd_adagrad',
	'rsub_adagrad'
]

stats = {}
for name in names:
	stats[name] = Profiler()


def clear(arr):
	with stats['clear']:
		arr.np *= 0


def copy(dst, src):
	with stats['copy']:
		np.copyto(dst.np, src.np)


def add(dst, one, two):
	with stats['add']:
		np.add(one.np, two.np, out=dst.np)


def radd(dst, arr):
	with stats['radd']:
		dst.np += arr.np


def clip(dst, src, lv, rv):
	with stats['clip']:
		np.clip(src.np, lv, rv, out=dst.np)


def rclip(dst, lv, rv):
	with stats['rclip']:
		np.clip(dst.np, lv, rv, out=dst.np)


def muls(dst, one, two):
	with stats['muls']:
		np.mul(one.np, two, out=dst.np)


def mul(dst, one, two):
	with stats['mul']:
		np.mul(one.np, two.np, out=dst.np)


def rmuls(dst, src):
	with stats['rmuls']:
		dst.np *= src


def rmul(dst, src):
	with stats['rmul']:
		dst.np *= src.np


def dot(dst, one, two):
	with stats['dot']:
		np.dot(one.np, two.np, out=dst.np)


def raddouter(dst, one, two):
	with stats['raddouter']:
		so = one.np.shape[0]
		st = two.np.shape[0]
		dst.np += np.dot(one.np.reshape(so, 1), two.np.reshape(1, st))
		# dst.np += np.outer(one.np, two.np)


def rsubmuls(dst, one, two):
	with stats['rsubmuls']:
		dst.np -= one.np*two


def rsubmul(dst, one, two):
	with stats['rsubmul']:
		dst.np -= one.np*two.np


def tanh(dst, src):
	with stats['tanh']:
		np.tanh(src.np, out=dst.np)


def bptanh(dst, err, out):
	with stats['bptanh']:
		dst.np = err.np*(1 - out.np**2)


def softmax(dst, src):
	with stats['softmax']:
		exp = np.exp(src.np)
		dst.np = exp/np.sum(exp)


def softmaxloss(dst, src, aim):
	with stats['softmaxloss']:
		np.subtract(src.np, aim.np, out=dst.np)
		return -np.log(np.dot(src.np, aim.np))


def radd_adagrad(dst, grad):
	with stats['radd_adagrad']:
		dst.np += grad.np**2


def rsub_adagrad(dst, grad, factor, rate):
	with stats['rsub_adagrad']:
		dst.np -= (factor/np.sqrt(rate.np))*grad.np
