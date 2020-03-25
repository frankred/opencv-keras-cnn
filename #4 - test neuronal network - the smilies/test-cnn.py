import numpy as np
import cv2
import tensorflow as tf

CATEGORIES = ["No smilie", "Happy Smilie", "Sad Smilie"]  
model = tf.keras.models.load_model("model.h5")
IMG_SIZE = 26
TEST_IMGS =  []

'''

def addPrepare(filepath):
        img = cv2.imread(filepath)
        img = cv2.Canny(img,33,76)
        img = np.resize(img, (-1, 26, 26, 1))
        TEST_IMGS.append(img)

addPrepare('001.jpg')
addPrepare('test-image-white.png')
addPrepare('test-image.png')

TEST_IMGS = np.asarray(TEST_IMGS)
TEST_IMGS = TEST_IMGS.astype('float32')
TEST_IMGS /= 255.0

prediction = model.predict([TEST_IMGS[2]])
print(prediction)q

'''

# Video Loop
cap = cv2.VideoCapture(0)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
## cv2.namedWindow('cropped_canny_frame', cv2.WINDOW_NORMAL)

img_out_counter = 0
ACTIVATE_CANNY = False

SMILING_IN_ROW = 0
SAD_IN_ROW = 0


while(True):
    if cv2.waitKey(1) & 0xFF == ord('c'):
        ACTIVATE_CANNY = True

    isSmiling = False
    isSad = False

    ret, frame = cap.read()
    
    cropped_frame = frame[200:226, 300:326].copy()
    cropped_canny_frame = cv2.Canny(cropped_frame,33,76)
    # cv2.imshow('cropped_canny_frame', cropped_canny_frame)
    
    cropped_canny_frame = np.resize(cropped_canny_frame, (26, 26, 1))
    cropped_canny_frame = cropped_canny_frame[np.newaxis, ...]
    prediction = model.predict(cropped_canny_frame)

    if prediction[0][2] == 1.0:
        isSad = True
        SAD_IN_ROW = SAD_IN_ROW+1
        SMILING_IN_ROW = 0

    if prediction[0][1] == 1.0:
        isSmiling = True
        SMILING_IN_ROW = SMILING_IN_ROW + 1
        SAD_IN_ROW = 0

    frame = cv2.rectangle(frame,(300, 200),  (326, 226), (255, 0, 0), 2)
    if isSmiling and SMILING_IN_ROW > 10:
        frame = cv2.rectangle(frame,(280, 180),  (346, 246), (9, 191, 0), 30)
    if isSad and  SAD_IN_ROW > 10:
        frame = cv2.rectangle(frame,(280, 180),  (346, 246), (0, 0, 192), 30)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        outpath = str(img_out_counter) + '.jpg'
        print('Save to file', outpath)
        cv2.imwrite('output/' + outpath, cropped_frame)
        img_out_counter = img_out_counter + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame = cv2.flip(frame, 1)
    if isSmiling and SMILING_IN_ROW > 10:
        frame = cv2.putText(frame, 'HAPPY', (50, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (9, 191, 0), 4, cv2.LINE_AA) 
    if isSad and  SAD_IN_ROW > 10:
        frame = cv2.putText(frame, 'SAD', (50, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 192), 4, cv2.LINE_AA) 


    cv2.imshow('frame',frame)

    
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()