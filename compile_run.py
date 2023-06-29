#!/usr/bin/env python3
import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np

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
    input = np.random.randint(0, 255, input_tensor.shape).astype("float32")
    
    # Run the inference
    outputs = sess.run(input)
    
    print("== Output ==")
    print(outputs)
    print(outputs[0].numpy())


if __name__ == "__main__":
    run_example()
