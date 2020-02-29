# -*- coding: utf-8 -*-


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxClassifier(nn.Module):
	'''
	Softmax classifier that will be trained via the generated visual feature
	for the final performance evaluation.
	'''
	def __init__(self, input_dim, n_class):
		super(SoftmaxClassifier, self).__init__()
		self.n_class = n_class

		self.cls = nn.Sequential(
			nn.Linear(input_dim, n_class),
		)
		
	def forward(self, x):
		x = self.cls(x)
		# x = F.softmax(x, dim=1)
		x = F.log_softmax(x, dim=1)
		return x