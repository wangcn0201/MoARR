from dartsutils import genotypes
from dartsutils.model import AddConcetCell
import torch.nn as nn


class NasNet_NC(nn.Module): ###NC3###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(NasNet_NC, self).__init__()
		#print(C_prev_prev, C_prev, C)
		genotype = eval("genotypes.%s" % "NASNet")
		concat = genotype.normal_concat
		C_prev_prev, C_prev, C = pre_input_channel, input_channel, (out_channel//len(concat))//2*2
		reduction, reduction_prev = False, reduction_prev
		self.Cell = AddConcetCell(genotype, C_prev_prev, C_prev, C, reduction, reduction_prev)
		self.out_dim = C*len(concat)
		self.depth = 3
		self.density_edge = 26
		self.density_node = 16

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out


class NasNet_RC(nn.Module): ###RC3###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(NasNet_RC, self).__init__()
		#print(C_prev_prev, C_prev, C)
		genotype = eval("genotypes.%s" % "NASNet")
		concat = genotype.reduce_concat
		C_prev_prev, C_prev, C = pre_input_channel, input_channel, (out_channel//len(concat))//2*2
		reduction, reduction_prev = True, reduction_prev
		self.Cell = AddConcetCell(genotype, C_prev_prev, C_prev, C, reduction, reduction_prev)
		self.out_dim = C*len(concat)
		self.depth = 5
		self.density_edge = 24
		self.density_node = 16

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out