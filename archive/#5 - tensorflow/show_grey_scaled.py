import numpy as np
import cv2
from matplotlib import pyplot as plt


'''


img = cv2.imread("validation/smilie-happy/060.jpg")
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
'''

width=180
height=180


img = cv2.imread("validation/smilie-happy/060.jpg",0)
cv2.imshow('ORIGINAL', cv2.resize(img, (width, height)))


'''
equ = cv2.equalizeHist(img)
cv2.imshow('HISTO EQU', cv2.resize(equ, (width, height)))
'''

def on_trackbar_thresh1(val_new):
    global valThres1
    valThres1 = val_new
    print(valThres1)
    print(valThres2)
    print('----')
    cv2.imshow('EDGES', cv2.resize(cv2.Canny(img,valThres1,valThres2), (width*2, height*2)))

def on_trackbar_thresh2(val_new):
    global valThres2
    valThres2 = val_new
    print(valThres1)
    print(valThres2)
    print('----')
    cv2.imshow('EDGES', cv2.resize(cv2.Canny(img,valThres1,valThres2), (width*2, height*2)))


thresholdSlider1 = 200
thresholdSlider2 = 200
valThres1 = 0
valThres2 = 0
cv2.namedWindow('EDGES')


trackbar_name = 'threshold1 %d' % thresholdSlider1
cv2.createTrackbar(trackbar_name, 'EDGES' , 0, thresholdSlider1, on_trackbar_thresh1)

trackbar_name2 = 'threshold2 %d' % thresholdSlider2
cv2.createTrackbar(trackbar_name2, 'EDGES' , 0, thresholdSlider2, on_trackbar_thresh2)


on_trackbar_thresh1(0)
on_trackbar_thresh2(0)
# Wait until user press some key
cv2.waitKey()


'''
clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)
cv2.imshow('CLAHE', cv2.resize(res_clahe, (width, height)))


# waiting for key event
cv2.waitKey(0)

# destroying all windows
cv2.destroyAllWindows()

# Reading color image

image_enhanced = cv2.equalizeHist(img)
plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
plt.show()


lab= cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)



limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)





# Converting color image to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,200,200) 
cv2.imshow('Edges',edges) 

# waiting for key event
cv2.waitKey(0)

# destroying all windows
cv2.destroyAllWindows()
'''