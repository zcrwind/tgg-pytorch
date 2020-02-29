# -*- coding: utf-8 -*-

'''models of adversarial unseen feature synthesis (AUFS)'''

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class _netG(nn.Module):
	'''Generator with noise z.'''
	def __init__(self, se_fea_dim, vi_fea_dim, z_dim):
		super(_netG, self).__init__()
		self.main = nn.Sequential(nn.Linear((z_dim + se_fea_dim), 2048),
                                  nn.LeakyReLU(),

                                  nn.Linear(2048, 4096),
                                  nn.LeakyReLU(),

                                  nn.Linear(4096, vi_fea_dim),
                                  nn.Tanh())

	def forward(self, se_fea, z):
		_input = torch.cat([z, se_fea], 1)
		output = self.main(_input)
		return output


class _netG2(nn.Module):
	'''Generator without noise z.'''
	def __init__(self, se_fea_dim, vi_fea_dim):
		super(_netG2, self).__init__()
		self.main = nn.Sequential(nn.Linear((se_fea_dim), 1024),
								  nn.BatchNorm1d(1024),
                                  nn.LeakyReLU(),

                                  nn.Linear(1024, 2048),
                                  nn.BatchNorm1d(2048),
                                  nn.LeakyReLU(),

                                  nn.Linear(2048, 4096),
                                  nn.BatchNorm1d(4096),
                                  nn.LeakyReLU(),

                                  nn.Linear(4096, vi_fea_dim))

	def forward(self, se_fea):
		output = self.main(se_fea)
		return output
		

class _netD(nn.Module):
	'''
		Discriminator
	'''
	def __init__(self, vi_fea_dim, n_class):
		'''n_class: number of the classes for auxiliary classification.'''
		super(_netD, self).__init__()
		self.D_shared = nn.Sequential(nn.Linear(vi_fea_dim, 4096),
									  nn.ReLU(),
									  nn.Linear(4096, 1024),
									  nn.ReLU())

		self.D_gan = nn.Linear(1024, 1)					# for the GAN loss
		self.D_classifier = nn.Linear(1024, n_class)	# for the auxiliary classification loss

	def forward(self, vi_feature):
		hidden = self.D_shared(vi_feature)
		gan_loss = self.D_gan(hidden)
		classification_loss = self.D_classifier(hidden)
		return gan_loss, classification_loss


class Regressor(nn.Module):
	'''
		Regressor for mapping the generated visual feature to
		the semantic feature (e.g., word vector or attribute)
	'''
	def __init__(self, input_dim, output_dim):
		'''
		Args:
			input_dim:  the dimension of generated visual feature (e.g., 2048 for resnet101 feature)
			output_dim: the dimension of semantic feature (e.g., 312 for CUB attribute)
		'''
		super(Regressor, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_dim, 4096),
			nn.LeakyReLU(),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.model(x)
		return x
		


if __name__ == '__main__':
	vi_fea_dim = 8192
	se_fea_dim = 300
	z_dim = 100
	netG = _netG(se_fea_dim, vi_fea_dim, z_dim)
	netD = _netD(vi_fea_dim)
	print(netG)
	print(netD)
