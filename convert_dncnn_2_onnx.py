#!/usr/bin/env python3
import os
import logging
import torch
from torchsummary import summary
from models.network_dncnn import IRCNN as ircnn
from pudb import set_trace
import argparse

parser = argparse.ArgumentParser(
                    prog='convert_dncnn_2_onnx',
                    description='convert to onnx',
                    epilog='output file with onnx suffix')

parser.add_argument('--bs', required=True, type=int)
args = parser.parse_args()
print(f'Requested batch size: {args.bs}')

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

n_channels = 3
img_dim = (n_channels, 256, 256)
input_shape = (args.bs, img_dim[0], img_dim[1], img_dim[2])


model_pool = 'model_zoo'             # fixed
model_name = 'dncnn_25'              #
nb = 17                              # what is this?
from models.network_dncnn import DnCNN as net
model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')


print(summary(model, img_dim))

# Input to the model
x = torch.randn(args.bs,
                img_dim[0], img_dim[1], img_dim[2],
                requires_grad=True)


torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  model_name + ".onnx",              # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
