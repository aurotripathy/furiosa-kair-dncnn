#### Example Furiosa Worklfow for Image Restoration

Uses the dncnn model from [KAIR](https://github.com/cszn/KAIR) and leverages portions of the code.
You can find the dncnn model paramters is the accompanying [model zoo](https://github.com/cszn/KAIR/tree/master/model_zoo) 


##### Download the dncnn model
Download the PyTorch model as instructed in the [model zoo folder](https://github.com/cszn/KAIR/tree/master/model_zoo)
```
python main_download_pretrained_models.py --models "DnCNN" --model_dir "model_zoo"
```
I'm working with the three-channel (RGB) model `dncnn_color_blind`

##### Convert to ONNX
```
./convert_dncnn_2_onnx.py --bs 1 --model_name dnn_color_blind
```
This generates an ONNX file named `dncnn_color_blind.onnx`

##### Quantize the Model
```
./quantize.py --model_name dncnn_color_blind.onnx
```
This generates an DFG file named `dncnn_color_blind.dfg`

##### Compile and run
```
./compile_run.py --quant_model_path dncnn_color_blind.dfg
```

##### Results
```
testsets/set12/01.png - PSNR: 24.85 dB; SSIM: 0.5036.
testsets/set12/02.png - PSNR: 24.59 dB; SSIM: 0.4453.
testsets/set12/03.png - PSNR: 24.67 dB; SSIM: 0.5327.
testsets/set12/04.png - PSNR: 24.69 dB; SSIM: 0.6447.
testsets/set12/05.png - PSNR: 24.62 dB; SSIM: 0.6095.
testsets/set12/06.png - PSNR: 24.65 dB; SSIM: 0.5332.
testsets/set12/07.png - PSNR: 24.84 dB; SSIM: 0.5501.
testsets/set12/08.png - PSNR: 24.59 dB; SSIM: 0.5315.
testsets/set12/09.png - PSNR: 24.62 dB; SSIM: 0.6321.
testsets/set12/10.png - PSNR: 24.61 dB; SSIM: 0.5944.
testsets/set12/11.png - PSNR: 24.60 dB; SSIM: 0.6002.
testsets/set12/12.png - PSNR: 24.60 dB; SSIM: 0.6262.

```
