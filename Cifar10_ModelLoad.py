import os
import sys
import time
import glob
import numpy as np
import torch
from dartsutils import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from CodeProcessor import CodeToCifarModel
from dartsutils.flops_counter import get_model_complexity_info

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../Data', help='location of the cifar data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--code', type=str, default='', help='which code to use')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--load_model_path', type=str, default='Cifar10Results/Code1_weights.pt', help='path of pretrained model dir')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    
    Code = eval(args.code)
    model = CodeToCifarModel(Code, num_classes=CIFAR_CLASSES, auxiliary=True)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model.load_state_dict(torch.load(args.load_model_path))
    logging.info("Model Loaded from : %s", str(args.load_model_path))

    input_size = (3, 32, 32)
    model.drop_path_prob = 0.0
    flops, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    logging.info("FLOPs size = %fGB", flops/1e9)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils.data_transforms_cifar10(args)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

