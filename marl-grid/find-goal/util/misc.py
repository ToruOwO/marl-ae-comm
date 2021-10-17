from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import os, time, cv2, numpy as np


class bcolors:
	HEADER = '\033[95m'
	b = blue = OKBLUE = '\033[94m'
	g = green = OKGREEN = '\033[92m'
	y = yellow = WARNING = '\033[93m'
	r = red = FAIL = '\033[91m'
	c = cyan = '\033[36m'
	lb = lightblue ='\033[94m'
	p = pink = '\033[95m'
	o = orange='\033[33m'
	p =pink='\033[95m'
	lc = lightcyan='\033[96m'
	end = ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def cprint(print_str, color=None, float_num=False, return_str=False):
	if float_num is not False:
		# make it colorful
		cmap = [31, 32, 33, 34, 35, 36, 37, 91, 92, 93, 94, 95, 96, 97]
		cmap_idx = int(float_num * (len(cmap) - 1)) # floor
		c = '\033[{}m'.format(cmap[cmap_idx])
	else:
		if not hasattr(bcolors, color):
			warnings.warn('Unknown color {}'.format(color))
			if return_str:
				return print_str
			print(print_string)
		else:
			c = getattr(bcolors, color)
	e = getattr(bcolors, 'end')
	c_str = '{}{}{}'.format(c, print_str, e)
	if return_str:
		return c_str
	print(c_str)
	return


def check_done(done):
	if type(done) is bool:
		return done
	elif type(done) is dict:
		return done['__all__']
	else:
		raise ValueError(f'unknown done signal {type(done)}')
