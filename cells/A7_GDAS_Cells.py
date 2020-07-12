from dartsutils import genotypes
from dartsutils.model import AddConcetCell
import torch.nn as nn


class GDAS_V1_NC(nn.Module): ###NC11###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(GDAS_V1_NC, self).__init__()

		genotype = eval("genotypes.%s" % "GDAS_V1")
		concat = genotype.normal_concat
		C_prev_prev, C_prev, C = pre_input_channel, input_channel, (out_channel//len(concat))//2*2
		reduction, reduction_prev = False, reduction_prev
		self.Cell = AddConcetCell(genotype, C_prev_prev, C_prev, C, reduction, reduction_prev)
		self.out_dim = C*len(concat)
		self.depth = 9
		self.density_edge = 20
		self.density_node = 13

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out


class GDAS_V1_RC(nn.Module): ###RC10###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(GDAS_V1_RC, self).__init__()

		genotype = eval("genotypes.%s" % "GDAS_V1")
		concat = genotype.reduce_concat
		C_prev_prev, C_prev, C = pre_input_channel, input_channel, (out_channel//len(concat))//2*2
		reduction, reduction_prev = True, reduction_prev
		self.Cell = AddConcetCell(genotype, C_prev_prev, C_prev, C, reduction, reduction_prev)
		self.out_dim = C*len(concat)
		self.depth = 5
		self.density_edge = 20
		self.density_node = 13

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out


class GDAS_V2_NC(nn.Module): ###NC12###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(GDAS_V2_NC, self).__init__()

		genotype = eval("genotypes.%s" % "GDAS_V2")
		concat = genotype.normal_concat
		C_prev_prev, C_prev, C = pre_input_channel, input_channel, (out_channel//len(concat))//2*2
		reduction, reduction_prev = False, reduction_prev
		self.Cell = AddConcetCell(genotype, C_prev_prev, C_prev, C, reduction, reduction_prev)
		self.out_dim = C*len(concat)
		self.depth = 5
		self.density_edge = 20
		self.density_node = 13

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out


class GDAS_V2_RC(nn.Module): ###RC11###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(GDAS_V2_RC, self).__init__()

		genotype = eval("genotypes.%s" % "GDAS_V2")
		concat = genotype.reduce_concat
		C_prev_prev, C_prev, C = pre_input_channel, input_channel, (out_channel//len(concat))//8*8
		reduction, reduction_prev = True, reduction_prev
		self.Cell = AddConcetCell(genotype, C_prev_prev, C_prev, C, reduction, reduction_prev)
		self.out_dim = C*len(concat)
		self.depth = 2
		self.density_edge = 8
		self.density_node = 5

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out

