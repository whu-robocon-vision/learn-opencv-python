import os
import sys
import cv2 as cv

def create_pos_n_neg():
    for file_type in ['pos']:
        for img in os.listdir(file_type):
            image = cv.imread("./pos/" + img)
            if(file_type=='neg'):
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
            elif (file_type == 'pos'):
                line = file_type + '/' + img + ' 1 0 0 ' + str(image.shape[1]) + ' ' + str(image.shape[0]) + '\n'
                with open('info.txt', 'a') as f:
                    f.write(line)
create_pos_n_neg()