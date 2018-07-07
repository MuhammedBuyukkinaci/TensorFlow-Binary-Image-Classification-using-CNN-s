# TensorFlow Binary Image Classification using CNN's
This is a binary image classification project using Convolutional Neural Networks and TensorFlow API (no Keras) on Python 3.
[Read all story in Turkish](https://medium.com/@mubuyuk51/tensorflow-i%CC%87le-i%CC%87kili-binary-resim-s%C4%B1n%C4%B1fland%C4%B1rma-69b15085f92c).
# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Notebook

Download .ipynb file from [here](https://github.com/MuhammedBuyukkinaci/My-Jupyter-Files-1/blob/master/tensorflow_binary_image_classification2.ipynb) and run:

```jupyter lab ``` or ```jupyter notebook ```

# Data
No MNIST or CIFAR-10.

This is a repository containing datasets of 6400 training images and 1243 testing images.No problematic image.

Data is in datasets.7z . Just extract train_data_bi.npy and test_data_bi.npy .

7z may not work on Linux and MacOS. You can download .rar extension version from [here](
https://www.dropbox.com/s/ezmsiz0p364shxz/datasets.rar?dl=0) or .zip extension version from [here](
https://www.dropbox.com/s/cx6f238aoxjem6j/datasets_zip.zip?dl=0).
It is 101 MB.

Extract files from datasets.rar. Then put it in TensorFlow-Image-Classification-Convolutional-Neural-Networks folder.
train_data_bi.npy is containing 6400 training photos with labels.

test_data_bi.npy is containing 1243 testing photos with labels.

Classes are table & glass.

Classes are equal(3200 glass - 3200 table). 

# Training
Training on GPU:

```python binary_image_classification_GPU.py ```

Training on CPU:

```python binary_image_classification_CPU.py ```

# CPU or GPU
I trained on GTX 1050. 1 epoch lasted 20 seconds approximately.

If you are using CPU, which I do not recommend, change the lines below:
```
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=config) as sess:
```
to
```
with tf.Session() as sess:
```
# Architecture

AlexNet is used as architecture. 5 convolution layers and 3 Fully Connected Layers with 0.5 Dropout Ratio. 60 million Parameters.
![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/alexnet_architecture.png) 

# Results
Trained 5 epochs. Accuracy, AUC and Loss graphs are below:

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/results.png)

# Predictions

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Binary-Image-Classification-using-CNN-s/blob/master/binary_preds.png)


