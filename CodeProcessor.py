from cells.A1_Darts_Cells import *
from cells.A2_NasNet_Cells import *
from cells.A3_AmoebaNet_Cells import *
from cells.A4_ASAP_Cells import *
from cells.A5_ENAS_Cells import *
from cells.A6_RENAS_Cells import *
from cells.A7_GDAS_Cells import *
from cells.A8_ShuffleNetV2_Cells import *
from dartsutils.model import AuxiliaryHeadCIFAR
import torch.nn as nn
from dartsutils.adaptive_avgmax_pool import SelectAdaptivePool2d
from dartsutils import utils
import copy

'''
Contents of a 'Code' that discribe a 'CNN Architecture' :
	0  [use stem cell, channel size for first NCs, channel size ratio of stem, number of following NCs, NC index]
	1  [RC index, channel size multiplier for NCs, channel size ratio of RC, number of following NCs, NC index]
	2  [RC index, channel size multiplier for NCs, channel size ratio of RC, number of following NCs, NC index]
	3  [RC index, channel size multiplier for RC, whether add _3rd RC, CC index, -1]
Ranges of Code Contents :
	0  [{1}, {28*4,32*4,36*4}, {0.75}, {4,5,6,7}, {0,...,9}]
	1  [{0,...,9}, {1.5,2,2.5}, {1}, {4,5,6,7}, {0,...,9}]
	2  [{0,...,9}, {1.5,2,2.5}, {1}, {4,5,6,7}, {0,...,9}]
	3  [{0,...,9}, {1.5,2,2.5}, {0,1}, {0,1,2}, -1]
'''
CellInfos = {
	'0,0': Darts_V1_NC, '0,1': Darts_V2_NC, '0,2': NasNet_NC, '0,3': AmoebaNet_NC,
	'0,4': ASAP_NC, '0,5': ENAS_NC, '0,6': RENAS_NC, '0,7': GDAS_V1_NC, 
	'0,8': GDAS_V2_NC, '0,9': ShuffleNetV2_NC,

	'1,0': Darts_V1_RC, '1,1': Darts_V2_RC, '1,2': NasNet_RC, '1,3': AmoebaNet_RC, 
	'1,4': ASAP_RC, '1,5': ENAS_RC, '1,6': RENAS_RC, '1,7': GDAS_V1_RC, 
	'1,8': GDAS_V2_RC, '1,9': ShuffleNetV2_RC, 

	'2,0': 'avg', '2,1': 'avgmax', '2,2': 'max'
}

def RandomCode_Generation():
	import random
	Code = []
	FirstPart = [0, random.randint(0,2), 0, random.randint(0,3), random.randint(0,9)]
	Code.append(FirstPart)
	SecondPart = [random.randint(0,9), random.randint(0,2), 0, random.randint(0,3), random.randint(0,9)]
	Code.append(SecondPart)
	ThirdPart = [random.randint(0,9), random.randint(0,2), 0, random.randint(0,3), random.randint(0,9)]
	Code.append(ThirdPart)
	ForthPart = [random.randint(0,9), random.randint(0,2), random.randint(0,1), random.randint(0,2), 0]
	Code.append(ForthPart)
	return Code

def model_info(model, dataname):
	from dartsutils.flops_counter import get_model_complexity_info
	if dataname in ["CIFAR-10", "CIFAR-100"]:
		input_size = (3, 32,32)
	else:
		input_size = (3, 224,224)
	model.drop_path_prob = 0.0
	flops, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
	params, macs, flops = params/1e6, flops/1e6, flops/1e6
	return params, macs, flops

def batchacc_info(model, dataname):
	from BatchAcc import BatchAccMain
	epoch1_results, eary_stop_results = BatchAccMain(model, dataname)
	return epoch1_results, eary_stop_results

def model_infos_all(model, dataname):
	_, _, flops = model_info(model, dataname=dataname)
	_infos = model._infos
	params = utils.count_parameters_in_MB(model)
	if dataname not in ["CIFAR-10", "CIFAR-100"]:
		eary_stop_results = None
		_infos['density_node'], _infos['density_edge'] = None, None
	else:
		_, eary_stop_results = batchacc_info(model,dataname)
	infos = {
		'params(MB)':params,
		'flops(MB)':flops,
		'density_edge': _infos['density_edge'],
		'density_node': _infos['density_node'],
		'depth':_infos['depth'],
		'final_size':_infos['final_size'],
		'eary_stop_results':eary_stop_results
	}
	return infos

class CodeToCifarModel(nn.Module):
	'''
	Contents of a 'Code' that discribe a 'CNN Architecture' :
		0  [use stem cell, channel size for first NCs, channel size ratio of stem, number of following NCs, NC index]
		1  [RC index, channel size multiplier for NCs, channel size ratio of RC, number of following NCs, NC index]
		2  [RC index, channel size multiplier for NCs, channel size ratio of RC, number of following NCs, NC index]
		3  [RC index, channel size multiplier for RC, whether add _3rd RC, CC index, -1]
	Ranges of Code Contents :
		0  [{1}, {28*4,32*4,36*4}, {0.75}, {4,5,6,7}, {0,...,9}]
		1  [{0,...,9}, {1.5,2,2.5}, {1}, {4,5,6,7}, {0,...,9}]
		2  [{0,...,9}, {1.5,2,2.5}, {1}, {4,5,6,7}, {0,...,9}]
		3  [{0,...,9}, {1.5,2,2.5}, {0,1}, {0,1,2}, -1]
	'''
	def __init__(self, Code, num_classes, auxiliary):
		super(CodeToCifarModel, self).__init__()
		FirstPart, SecondPart, ThirdPart, ForthPart = Code
		INFOs = {
			'first_channel' : [28*4, 32*4, 36*4],
			'stem_ratio' : [0.75],
			'number' : [4,5,6,7],
			'channel_multiplier': [1.5, 2, 2.5],
			'channel_ratio' : [1]
		}

		self._cells = nn.ModuleList()
		self._infos = {
			'density_edge':0,
			'density_node':0,
			'depth': 0,
			'final_size': 0}

		# FirstPart
		_, first_channel, _, _1stnumber, _1stNCname = FirstPart
		stem = 1
		self._channel = INFOs['first_channel'][first_channel]
		self.stem = stem
		if stem:
			self.stemcell = nn.Sequential(
					nn.Conv2d(3, int(self._channel*INFOs['stem_ratio'][0]), 3, padding=1, bias=False),
					nn.BatchNorm2d(int(self._channel*INFOs['stem_ratio'][0]))
					)
			pre_input_channel = input_channel = int(self._channel*INFOs['stem_ratio'][0])
			self._infos['depth'] = 1
			self._infos['density_edge'] = 1
			self._infos['density_node'] = 1
		else:
			pre_input_channel = input_channel = 3
		for i in range(INFOs['number'][_1stnumber]):
			cell = CellInfos['0,'+str(_1stNCname)](pre_input_channel=pre_input_channel, input_channel=input_channel, out_channel=int(self._channel))
			pre_input_channel, input_channel = input_channel, cell.out_dim
			self._cells += [cell]
			self._infos['depth'] += cell.depth
			self._infos['density_edge'] += cell.density_edge
			self._infos['density_node'] += cell.density_node

		# SecondPart
		_1stRCname, _1stchannel_multiplier, _, _2ndnumber, _2ndNCname = SecondPart
		self._channel = self._channel * INFOs['channel_multiplier'][_1stchannel_multiplier]
		cell = CellInfos['1,'+str(_1stRCname)](pre_input_channel=pre_input_channel, input_channel=input_channel, out_channel=int(self._channel*INFOs['channel_ratio'][0]))
		pre_input_channel, input_channel = input_channel, cell.out_dim
		self._cells += [cell]
		self._infos['depth'] += cell.depth
		self._infos['density_edge'] += cell.density_edge
		self._infos['density_node'] += cell.density_node
		for i in range(INFOs['number'][_2ndnumber]):
			reduction_prev = True if i == 0 else False
			cell = CellInfos['0,'+str(_2ndNCname)](pre_input_channel=pre_input_channel, input_channel=input_channel, out_channel=int(self._channel), reduction_prev=reduction_prev)
			pre_input_channel, input_channel = input_channel, cell.out_dim
			self._cells += [cell]
			self._infos['depth'] += cell.depth
			self._infos['density_edge'] += cell.density_edge
			self._infos['density_node'] += cell.density_node

		# ThirdPart
		_2ndRCname, _2ndchannel_multiplier, _, _3rdnumber, _3rdNCname = ThirdPart
		self._channel = self._channel * INFOs['channel_multiplier'][_2ndchannel_multiplier]
		cell = CellInfos['1,'+str(_2ndRCname)](pre_input_channel=pre_input_channel, input_channel=input_channel, out_channel=int(self._channel*INFOs['channel_ratio'][0]))
		pre_input_channel, input_channel = input_channel, cell.out_dim
		self._cells += [cell]
		self._infos['depth'] += cell.depth
		self._infos['density_edge'] += cell.density_edge
		self._infos['density_node'] += cell.density_node
		if auxiliary:
			self.auxiliary_head = AuxiliaryHeadCIFAR(input_channel, num_classes)
			self.auxiliary_head_index = len(self._cells)-1
		else:
			self.auxiliary_head = self.auxiliary_head_index = None
		for i in range(INFOs['number'][_3rdnumber]):
			reduction_prev = True if i == 0 else False
			cell = CellInfos['0,'+str(_3rdNCname)](pre_input_channel=pre_input_channel, input_channel=input_channel, out_channel=int(self._channel), reduction_prev=reduction_prev)
			pre_input_channel, input_channel = input_channel, cell.out_dim
			self._cells += [cell]
			self._infos['depth'] += cell.depth
			self._infos['density_edge'] += cell.density_edge
			self._infos['density_node'] += cell.density_node
		self._infos['final_size'] = 8

		# ForthPart
		_3rdRCname, _3rdchannel_multiplier, _3rdRC, CCname, _ = ForthPart
		self._channel = self._channel * INFOs['channel_multiplier'][_3rdchannel_multiplier]
		if _3rdRC:
			cell = CellInfos['1,'+str(_3rdRCname)](pre_input_channel=pre_input_channel, input_channel=input_channel, out_channel=int(self._channel))
			pre_input_channel, input_channel = input_channel, cell.out_dim
			self._cells += [cell]
			self._infos['depth'] += cell.depth
			self._infos['density_edge'] += cell.density_edge
			self._infos['density_node'] += cell.density_node
			self._infos['final_size'] = 4
		self.relu = nn.ReLU()
		self.global_pool = SelectAdaptivePool2d(pool_type=CellInfos["2,"+str(CCname)])
		self.last_linear = nn.Linear(input_channel * self.global_pool.feat_mult(), num_classes)
		return

	def forward(self, input):
		logits_aux = None
		s0 = s1 = input
		if self.stem:
			s0 = s1 = self.stemcell(s1)

		for i in range(len(self._cells)):
			s0, s1 = s1, self._cells[i](s0, s1, self.drop_path_prob)
			if self.auxiliary_head_index != None and i == self.auxiliary_head_index:
				logits_aux = self.auxiliary_head(s1)

		x = self.relu(s1)
		x = self.global_pool(x).flatten(1)
		logits = self.last_linear(x)
		return logits, logits_aux

