import sys
import cv2 as cv
import re
import numpy as np

filename = sys.argv[1]
imagename = filename.split('.')[0]+".JPEG"

ori_img = cv.imread(imagename)

origin = open(filename, 'r')

if ori_img is None:
    print("file not found:" + imagename)
    exit()

for line in origin.readlines():
    result = re.search("<xmin>([0-9]*)", line)
    if result:
        xmin = np.int(result.group(1))
    result = re.search("<xmax>([0-9]*)", line)
    if result:
        xmax = np.int(result.group(1))
    result = re.search("<ymin>([0-9]*)", line)
    if result:
        ymin = np.int(result.group(1))
    result = re.search("<ymax>([0-9]*)", line)
    if result:
        ymax = np.int(result.group(1))

cv.imwrite("/home/way/Learn/opencv/python/rugby/"+filename.split('.')[0] + "_modify.JPEG", ori_img[ymin:ymax, xmin:xmax])

cv.waitKey(0)