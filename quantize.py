#!/usr/bin/env python3

"""
Needs the imagnet 'val' folder with the structure retained 
and a few images for the calibration functions to work
"""
import sys

import onnx
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm
import glob
import os
from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
import cv2
import numpy as np
from pudb import set_trace
import argparse

parser = argparse.ArgumentParser(
                    prog='convert_dncnn_2_onnx',
                    description='convert to onnx',
                    epilog='output file with onnx suffix')

parser.add_argument('--model_name', required=True, type=str)
args = parser.parse_args()

img_dim = (3, 256, 256)
BATCH_SIZE = 1

class CalibrationDataset(Dataset):
	def __init__(self, imgs_path):
                self.imgs_path = imgs_path
                file_list = glob.glob(os.path.join(self.imgs_path, "*"))
                print(f'Printing calibration file list:')
                print(file_list)
                self.data = []
                for path in file_list:
                        self.data.append(path)
                print(self.data)
                self.img_dim = (img_dim[1], img_dim[2])
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
                img_path = self.data[idx]
                img = cv2.imread(img_path)
                # set_trace()
                img = cv2.resize(img, self.img_dim)
                img_tensor = torch.from_numpy(img)
                img_tensor = img_tensor.permute(2, 0, 1)
                return img_tensor



def create_quantized_dfg():

    model = onnx.load_model(args.model_name)

    calibration_dataset = CalibrationDataset('testsets/set12')

    calibration_dataloader = torch.utils.data.DataLoader(calibration_dataset,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True)
    model = optimize_model(model)

    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

    for calibration_data in tqdm.tqdm(calibration_dataloader,
                                         desc="Calibration",
                                         unit="images",
                                         mininterval=0.5):
        print(f'calibration data shape: {calibration_data.shape}')
        calibrator.collect_data([[np.float32(calibration_data.numpy())]])

    ranges = calibrator.compute_range()
    print(f'ranges" {ranges}')

    model_quantized = quantize(model, ranges)

    with open(os.path.splitext(args.model_name)[0] + '.dfg', "wb") as f:
        f.write(bytes(model_quantized))


if __name__ == "__main__":
    sys.exit(create_quantized_dfg())    
