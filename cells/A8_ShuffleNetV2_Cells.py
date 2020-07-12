from dartsutils.model import ShuffleV2Cell
import torch.nn as nn


class ShuffleNetV2_NC(nn.Module): ###NC13###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(ShuffleNetV2_NC, self).__init__()

		in_channels, out_channels, stride = input_channel, (out_channel//2)*2, 1
		self.Cell = ShuffleV2Cell(in_channels, out_channels, stride)
		self.out_dim = out_channels
		self.depth = 2
		self.density_edge = 3
		self.density_node = 3

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out


class ShuffleNetV2_RC(nn.Module): ###RC12###
	def __init__(self, input_channel, pre_input_channel, out_channel, reduction_prev=False):
		super(ShuffleNetV2_RC, self).__init__()

		in_channels, out_channels, stride = input_channel, (out_channel//2)*2, 2
		self.Cell = ShuffleV2Cell(in_channels, out_channels, stride)
		self.out_dim = out_channels
		self.depth = 2
		self.density_edge = 3
		self.density_node = 3

	def forward(self, s0, s1, drop_prob):
		out = self.Cell(s0, s1, drop_prob)
		return out
