#!/usr/bin/python3

from time import clock


class Profiler:
		def __init__(self):
			self.start = 0
			self.time = 0

		def __enter__(self):
			self.start = clock()

		def __exit__(self, *args):
			self.time += clock() - self.start
