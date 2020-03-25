![OpenCV + Tensorflow + Keras](/banner.png)

# Handwritten smilies detection (CNN)
This repository contains 4 python scripts that points out what a kernel is, how to create testdata for a convolutional neuronal network, how to create and train a neuronal network and finally how to run camera capture live images against this neuronal network.

# Training data
Image dimension is 26x26 and there are 3 classification types:

## Happy smilies (132 images)
![Happy smilies](/train-happy.png)

## Sad smilies (112 images)
![Sad smilies](/train-sad.png)

## No smilies (142 images)
![No smilies](/train-none.png)


# Validation data
## Happy smilies (15 images)
![Happy smilies](/validation-happy.png)

## Sad smilies (12 images)
![Sad smilies](/validation-sad.png)

## No smilies (16 images)
![No smilies](/validation-none.png)

The total amount of images is just 429 (including 43 images for validation). This is very little training data for a CNN, but the results are very good.


# Dependencies
## Python
python-3.6.8-amd64.exe https://www.python.org/downloads/release/python-368/

## Python Libraries
```
pip install opencv-python
pip install matplotlib
pip install tensorflow==1.8.0
pip install keras==2.1.5
pip install sklearn
pip install pandas
pip install absl-py
pip install pathlib
```

# Sources
https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/
https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
https://github.com/spmallick/learnopencv/tree/master/KerasCNN-CIFAR
https://www.tensorflow.org/tutorials/images/classification
https://github.com/spmallick/learnopencv
https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/
