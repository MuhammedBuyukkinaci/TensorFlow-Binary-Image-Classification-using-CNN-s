# TensorFlow-Image-Classification-Convolutional-Neural-Networks

This is a repository containing datasets of 6400 training images and 1243 testing images. 

Classes are table/glass.

Classes are equal(3200 glass -3200 table). 

There is no high-level API like Keras or TFLearn. Just TensorFlow.

There is no problematic image in training and testing dataset.

Download dataset from the link below. It is 101 MB:

https://www.dropbox.com/s/ezmsiz0p364shxz/datasets.rar?dl=0

Extract files from datasets.rar . Then put it in TensorFlow-Image-Classification-Convolutional-Neural-Networks .

train_data_bi.npy is containing 6400 training photos with labels.

test_data_bi.npy is containing 1243 testing photos with labels.

Accuracy score reached 90 percent on CV after 50 epochs.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/accuracy.png)

Cross entropy loss is plotted below.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/loss.png)

The architecture used in CNN's is below:

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/ARCHITECTURE.png)

A sample photo of glass:

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/glass_16.jpg)

A sample photo of table:

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/table_1488.jpg)





