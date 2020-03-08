import sys
import cv2 as cv
import numpy as np

SAME_IMAGE_DEPTH_AS_ORIGINAL = -1

def run():
    # Load Image
    src = cv.imread(cv.samples.findFile('lena.jpg'), cv.IMREAD_COLOR)
    

    ind = 0
    while True:

        ############################ Example 1 ############################
        kernel_original = np.array([
            [0,0,0], 
            [0,1,0], 
            [0,0,0]
        ])

        dst = cv.filter2D(src, SAME_IMAGE_DEPTH_AS_ORIGINAL, kernel_original)
        cv.imshow('Preview Window 1', dst)







        ############################ Example 2 ############################

        kernel_sharpen = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1]
        ])

        dst = cv.filter2D(src, SAME_IMAGE_DEPTH_AS_ORIGINAL, kernel_sharpen)
        cv.imshow('Preview Window 2', dst)

        
        ############################ Example 3 ############################
        # 1/9 = 0.1111
        kernel_blur = np.array([
            [0.1111,0.1111,0.1111],
            [0.1111,0.1111,0.1111],
            [0.1111,0.1111,0.1111]
        ])

        dst = cv.filter2D(src, SAME_IMAGE_DEPTH_AS_ORIGINAL, kernel_blur)
        cv.imshow('Preview Window 3', dst)

        dst2 = cv.filter2D(dst, SAME_IMAGE_DEPTH_AS_ORIGINAL, kernel_blur)
        dst3 = cv.filter2D(dst2, SAME_IMAGE_DEPTH_AS_ORIGINAL, kernel_blur)
        cv.imshow('Preview Window 4', dst3)

        
        ############################ Example 4 ############################
        kernel_edges = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ])
        dst = cv.filter2D(src, SAME_IMAGE_DEPTH_AS_ORIGINAL, kernel_edges)
        cv.imshow('Preview Window 5', dst)

        c = cv.waitKey()
        if c:
            break
        ind += 1
    return 0


if __name__ == '__main__':
    run()