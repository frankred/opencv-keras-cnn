import numpy as np
import cv2
import pickle
import joblib

pickle_in=open("latest-model.p","rb")  ## rb = READ BYTE
#model=joblib.load(open("latest-model.p", 'rb')) # 
pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'No smilie'
    elif classNo == 1: return 'Happy Smilie'
    elif classNo == 2: return 'Sad Smilie'


imgOrignal = cv2.imread('x.jpg')

# READ IMAGE
#success, imgOrignal = cap.read()

# PROCESS IMAGE
img = np.asarray(imgOrignal)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
cv2.imshow("Processed Image", img)
img = img.reshape(1, 32, 32, 1)
cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# PREDICT IMAGE
predictions = model.predict(img)
classIndex = model.predict_classes(img)
probabilityValue =np.amax(predictions)
if probabilityValue > threshold:
    #print(getCalssName(classIndex))
    cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("Result", imgOrignal)
