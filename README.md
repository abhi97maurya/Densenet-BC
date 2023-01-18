######## CSCE 636 Project #############
In this project, deep learning model is designed and implemented to perform
10-class image classification on the CIFAR-10 dataset.

For image classification on CIFAR-10-dataset, baseline model of DenseNet-BC architecture is used.
Pytorch Version used: 1.7.0

In the Densenet-BC architecture, total 190 layer is used with the growth rate of 40. In terms of layers
blocks, 3 dense-block layers and 2 transition layers are used.

In the Dense block implementation, in a single dense-block 31 bottleneck blocks are used.

Command for Training:

python main.py --mode train

Command for testing Public dataset:

python main.py --mode test

Command for testing Private dataset:

python main.py --mode predict

## Result of private dataset is stored in directory: 
Directory: "result_dir"
filename: "predictions.npy"