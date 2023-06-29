#!/usr/bin/env python3
import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np
import cv2
from utils import utils_image as util
from pudb import set_trace

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

quantized_model_path = './dncnn.dfg'
def run_example():
    runtime.__full_version__

    sess = session.create(str(quantized_model_path))
    print('Session Summary:')
    sess.print_summary()
    
    input_tensor = sess.inputs()[0]
    
    img_path = 'testsets/set12/05.png'
    img = cv2.imread(img_path)

    # degrade
    noise_level_img = 50
    img = util.uint2single(img)  # scale down [0,1]
    np.random.seed(seed=0)  # for reproducibility
    img += np.random.normal(0, noise_level_img/255., img.shape)
    util.imshow(util.single2uint(img),
                title='Noisy image {}'.format(noise_level_img))

    img = util.single2uint(img)
    # set_trace()
    img = img.transpose(2, 0, 1).astype("float32")
    input = img[np.newaxis, :, :, :]
    # set_trace()

    # Run the inference
    outputs = sess.run(input)
    
    print("== Output ==")
    print(f'Tensor output shape:\n{outputs}')

    out_img = np.squeeze(outputs[0].numpy()).transpose(1,2,0)
    cv2.imwrite('denoised.png', out_img)
    # util.imshow(out_img.round().astype(np.uint8), title='denoised')
    util.imshow(out_img.round().astype(np.uint8), title='denoised')
                
    
if __name__ == "__main__":
    run_example()
