import sys
import cv2 as cv
import os
import numpy as np

SAME_IMAGE_DEPTH_AS_ORIGINAL = -1


def run():
    source_image_path = 'resources/no-smilies.jpg'
    target_folder = 'smilie-no-smilie'

    moving_size_slow = 1
    moving_size_fast = 10
    rect_x = 400
    rect_y = 300
    rect_w = 26
    rect_h = 26
    rect_thickness = 1
    rect_border_color = (255, 0, 0)
    target_counter = 1

    source_image = cv.imread(cv.samples.findFile(source_image_path), cv.IMREAD_COLOR)
    cv.rectangle(source_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), rect_border_color, rect_thickness)

    while(True):
        c = cv.waitKeyEx()
        print(c)

        if (c == 2555904):
            rect_x = rect_x + moving_size_slow

        if (c == 2621440):
            rect_y = rect_y + moving_size_slow

        if (c == 2424832):
            rect_x = rect_x - moving_size_slow

        if (c == 2490368):
            rect_y = rect_y - moving_size_slow

        if (c == 56 or c == 119):
            rect_y = rect_y - moving_size_fast

        if (c == 54 or c == 100):
            rect_x = rect_x + moving_size_fast

        if (c == 50 or c == 115):
            rect_y = rect_y + moving_size_fast

        if (c == 52  or c == 97):
            rect_x = rect_x - moving_size_fast

        if (c == 27):
            break


        if(c == 13 or c == 32):
            source_image = cv.imread(cv.samples.findFile(source_image_path), cv.IMREAD_COLOR)
            crop_img = source_image[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
            cv.imwrite(os.path.join(target_folder, str(target_counter) + ".png"), crop_img)
            target_counter = target_counter + 1

        source_image = cv.imread(cv.samples.findFile(source_image_path), cv.IMREAD_COLOR)
        cv.rectangle(source_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), rect_border_color, rect_thickness)
        cv.imshow('traindata', source_image)


if __name__ == '__main__':
    run()
