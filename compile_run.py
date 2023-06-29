#!/usr/bin/env python3
import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np
import cv2
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
    
    # Generate the random input tensor according to the input shape
    img_path = 'testsets/set12/01.png'
    img = cv2.imread(img_path)
    set_trace()
    img = img.transpose(2, 0, 1).astype("float32")
    input = img[np.newaxis, :, :, :]
    
    # Run the inference
    outputs = sess.run(input)
    
    print("== Output ==")
    print(outputs)
    print(outputs[0].numpy())


if __name__ == "__main__":
    run_example()
