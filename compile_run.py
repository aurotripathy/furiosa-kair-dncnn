#!/usr/bin/env python3
import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np
import cv2
from utils import utils_image as util
from pudb import set_trace

import argparse

parser = argparse.ArgumentParser(
                    prog='convert_dncnn_2_onnx',
                    description='convert to onnx',
                    epilog='output file with onnx suffix')

parser.add_argument('--quant_model_path', required=True, type=str)
parser.add_argument('--noise_level_img', type=int, default=15, help='noise level: 15, 25, 50')
parser.add_argument('--show_img', type=bool, default=False, help='show the image')
args = parser.parse_args()

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

img_size = (256, 256)

def run_example():
    runtime.__full_version__

    sess = session.create(str(args.quant_model_path))
    print('Session Summary:')
    sess.print_summary()
    
    input_tensor = sess.inputs()[0]

    img_paths = util.get_image_paths('testsets/set12/')

    for img_path in img_paths:
        
        in_img = cv2.imread(img_path)
        in_img = cv2.resize(in_img, img_size)
        set_trace()

        # degrade
        img = util.uint2single(in_img)  # scale down [0,1]
        np.random.seed(seed=0)  # for reproducibility
        img += np.random.normal(0, args.noise_level_img/255., img.shape)
        if args.show_img:
            util.imshow(util.single2uint(img),
                        title='Noisy image {}'.format(args.noise_level_img))

        img = util.single2uint(img)
        img = img.transpose(2, 0, 1).astype("float32")
        input = img[np.newaxis, :, :, :]

        outputs = sess.run(input)  # inference

        # print("== Output ==")
        # print(f'Tensor output shape:\n{outputs}')

        out_img = np.squeeze(outputs[0].numpy()).transpose(1,2,0)
        cv2.imwrite('denoised.png', out_img)
        if args.show_img:
            util.imshow(out_img.round().astype(np.uint8), title='denoised')

        psnr = util.calculate_psnr(out_img, in_img, border=0)
        ssim = util.calculate_ssim(out_img, in_img, border=0)
        print(f'{img_path:s} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}.')


if __name__ == "__main__":
    run_example()
