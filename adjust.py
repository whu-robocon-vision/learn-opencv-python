import glob
import sys
import cv2 as cv
import numpy as np


ah = 0
aw = 0
ai = 0

files = glob.glob("./pos/*.png")
for file in files:
    img = cv.imread(file)
    h, w, ch = img.shape
    ah = ah + h
    aw = aw + w
    ai = ai + h / w
    # if h > w:
    #     img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    #     cv.imwrite(file, img)
    # img = cv.resize(img, (np.int(w * 0.1), np.int(h * 0.1)))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(file, img)
# ah = ah / len(files)
# aw = aw / len(files)
# ai = ai / len(files)

# for file in files:
#     img = cv.imread(file)
#     h, w, ch = img.shape
#     h = np.int(w * ai)
#     img = cv.resize(img, (w, h))
#     cv.imwrite(file, img)

print(ah, aw, ai)


