Code to reproduce the GAN method results for the AFHQ Wild and ImageNet1K data

prerequisites:

(-) The stylegan2ada pytorch repo https://github.com/NVlabs/stylegan2-ada-pytorch needs to be in the pythonpath
(-) The BigGAN pytorch repo https://github.com/ajbrock/BigGAN-PyTorch needs to be in the pythonpath and the model checkpoint needs to be at weights/138k/
(-) asserts need to be globally disabled (the stylegan2ada code threw an assert when stitching into specific layers)

--------------------------------------------------------------------------------------------------------------------

ckpts/pretrained/ contains trained 1x1 stitching convolutions

"evaluation_metrics_gan.py" reproduces the results for the GAN method shown in table 1 and 2,
which will be printed to the console

"measure_eval_time_gan.py" measures the runtime for the GAN method over 10 runs and prints
the mean and standard deviation to the console

"plot_samples.py" reconstructs 8 samples from imagenet validation and afhq validation when mapping through
a trained 1x1 stitching convolution from layer1 to layer4 into stylegan2 or from layer1 to layer3 into BigGAN

"train.py" trains for 1x1 convolutions to stitch resnet50 and stylegan2 or BigGAN, those will be saved to ckpts/

"settings.py": here the path for the AFHQ Wild validation data (and training data for using "train.py"),
and for the ImageNet1K validation data needs to be set
