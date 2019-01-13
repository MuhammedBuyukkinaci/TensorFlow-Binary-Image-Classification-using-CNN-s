# TensorFlow Binary Image Classification using CNN's
This is a binary image classification project using Convolutional Neural Networks and TensorFlow API (no Keras) on Python 3.
[Read all story in Turkish](https://medium.com/@mubuyuk51/tensorflow-i%CC%87le-i%CC%87kili-binary-resim-s%C4%B1n%C4%B1fland%C4%B1rma-69b15085f92c).

It is a ready-to-run code.
# Dependencies

```pip3 install -r requirements.txt```

# Notebook

```jupyter lab Binary_classification.ipynb ``` or ```jupyter notebook Binary_classification.ipynb ```

# Data
No MNIST or CIFAR-10.

This is a repository containing datasets of 5000 training images and 1243 testing images.No problematic image.

Data is in datasets.7z . Just extract train_data_bi.npy and test_data_bi.npy .

train_data_bi.npy is containing 5000 training photos with labels.

test_data_bi.npy is containing 1243 testing photos with labels.

Classes are table & glass. Classes are equal.

Download pure data from [here](https://www.kaggle.com/mbkinaci/glasses-tables). Warning 1.4 GB.

# Training
Training on GPU:

```python3 binary_image_classification_GPU.py ```

Training on CPU:

```python3 binary_image_classification_CPU.py ```

# Architecture

AlexNet is used as architecture. 5 convolution layers and 3 Fully Connected Layers with 0.5 Dropout Ratio. 60 million Parameters.
![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/alexnet_architecture.png) 

# Results
Trained 5 epochs. Accuracy, AUC and Loss graphs are below:

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/results.png)

# Predictions

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Binary-Image-Classification-using-CNN-s/blob/master/binary_preds.png)


