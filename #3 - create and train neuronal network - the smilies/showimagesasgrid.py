import itertools
import cv2
import os
import numpy as np

#User defined variables
dirname = "validation/smilie-sad" #Name of the directory containing the images
margin = 4 #Margin between pictures in pixels
w = 16 # Width of the matrix (nb of images)
h = 9 # Height of the matrix (nb of images)
n = w*h

filename_list = []

for file in os.listdir(dirname):
    if file.endswith(".jpg"):
        filename_list.append(file)

filename_list.sort()

print(filename_list)

imgs = [cv2.imread(os.getcwd()+"/"+dirname+"/"+file) for file in filename_list]

#Define the shape of the image to be replicated (all images should have the same shape)
img_h, img_w, img_c = imgs[0].shape

#Define the margins in x and y directions
m_x = margin
m_y = margin

#Size of the full size image
mat_x = 2*m_x + img_w * w + m_x * (w - 1)
mat_y = 2*m_y + img_h * h + m_y * (h - 1)

#Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
imgmatrix.fill(255)

#Prepare an iterable with the right dimensions
positions = itertools.product(range(h), range(w))

for (y_i, x_i), img in zip(positions, imgs):
    x = m_x +x_i * (img_w + m_x)
    y = m_y + y_i * (img_h + m_y)
    imgmatrix[y:y+img_h, x:x+img_w, :] = img


cv2.imshow('Smilies', imgmatrix)
cv2.waitKey(0)
