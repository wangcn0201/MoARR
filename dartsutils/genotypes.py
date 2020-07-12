from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'max_pool_3x3_bn'
    'avg_pool_3x3',
    'avg_pool_3x3_bn'
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7'
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_7x1_1x7',
    'conv_3x1_1x3s'
]

DARTS_V1 = Genotype(
  normal=[
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('skip_connect', 2)
    ], 
  normal_concat=[2, 3, 4, 5], 
  reduce=[
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 1), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('skip_connect', 2), 
    ('avg_pool_3x3', 0)
    ], 
  reduce_concat=[2, 3, 4, 5]
)

DARTS_V2 = Genotype(
  normal=[
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('skip_connect', 0), 
    ('dil_conv_3x3', 2)
    ], 
  normal_concat=[2, 3, 4, 5], 
  reduce=[
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 1), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 1), 
    ('max_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 1)
    ], 
  reduce_concat=[2, 3, 4, 5]
)

DARTS = DARTS_V2

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [0, 2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [3, 4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

GDAS_V1 = Genotype(
    normal=[
        ('skip_connect', 0),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 3),
        ('skip_connect', 0),
        ('sep_conv_5x5', 4),
        ('sep_conv_3x3', 3)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 2),
        ('sep_conv_5x5', 1),
        ('dil_conv_5x5', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_5x5', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

GDAS_V2 = Genotype(
    normal=[
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 2),
        ('skip_connect', 0),
        ('sep_conv_3x3', 2),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('sep_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('conv_3x1_1x3', 1),
        ('none', 1),
        ('max_pool_3x3_bn', 1),
        ('none', 1),
        ('conv_3x1_1x3', 0),
        ('none', 0),
        ('max_pool_3x3_bn', 0),
        ('none', 0)
    ],
    reduce_concat=[2, 3, 4, 5]
)

ENAS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_5x5', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0)
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 4),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 5),
        ('sep_conv_5x5', 0)
    ],
    reduce_concat=[2, 3, 6]
)

RENAS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('max_pool_3x3', 0),
        ('sep_conv_3x3', 3),
        ('skip_connect', 3)
    ],
    normal_concat=[2, 4, 5, 6],
    reduce=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('max_pool_3x3', 0),
        ('sep_conv_3x3', 3),
        ('skip_connect', 3)
    ],
    reduce_concat=[2, 4, 5, 6]
)

ASAP = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('skip_connect', 1)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0)
    ],
    reduce_concat=[2, 3, 4, 5]
)
