import cv2 as cv
import numpy as np


def pyramid_demo(img):
    level = 3
    temp = img.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images


def threshold_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("threshold value %s"%(ret))
    cv.imshow("binary", binary)


def local_threshold_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("local binary", binary)

def back_projection_demo():
    sample = cv.imread("/home/way/Learn/opencv/python/sample.png")
    target = cv.imread("/home/way/Learn/opencv/python/target.png")
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    # show images
    cv.imshow("sample", sample)
    cv.imshow("target", target)

    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 48], [0, 180, 0, 256])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow("backProjectionDemo", dst)


def video_demo():
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow("video", frame)
        if cv.waitKey(50) == 27:
            break


def pixel_access(img):
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    for row in range(height):
        for col in range(width):
            for ch in range(channels):
                img[row, col, ch] = 255 - img[row, col, ch]
    cv.imshow("pixels_demo", img)


def inverse(img):
    img = cv.bitwise_not(img)
    cv.imshow("inverse", img)


def getImgInfo(src):
    print(type(src))
    print(src.shape)
    print(src.size)
    print(src.dtype)
    pixel_img = np.array(src)
    print('src', src)
    print('pixel_img', pixel_img)

def create_img():
    '''
    img = np.zeros([400, 400, 3], np.uint8)
    img[ : , : , 0] = np.ones([400, 400]) * 255
    img[ : , : , 2] = np.ones([400, 400]) * 255
    cv.imshow("new image", img)
    '''

    img = np.ones([400, 400, 1], np.uint8)
    img = img * 127
    cv.imshow("new image", img)
    

def color_space_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)

def extrace_object_demo():
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        if ret == False:
            break

        lower_hsv = np.array([0, 43, 46])
        upper_hsv = np.array([10, 255, 255])
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)

        dst = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow("video", frame)
        cv.imshow("mask", dst)
        if cv.waitKey(40) == 27:
            break
        
def spilt_channel_demo():
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        if ret == False:
            break

        b, g, r = cv.split(frame)
        cv.imshow("frame", frame)
        cv.imshow("blue", b)
        cv.imshow("green", g)
        cv.imshow("red", r)

        frame[:, : , 0] = 0
        cv.imshow("changed channel", frame)
        frame = cv.merge([b, g, r])
        cv.imshow("merge channels", frame)
        if cv.waitKey(40) == 27:
            break

def laplace_demo(img):
    pyramid_imgs = pyramid_demo(img)
    level = len(pyramid_imgs)
    for i in range(level - 1, 0, -1):
        expand = cv.pyrUp(pyramid_imgs[i], dstsize=pyramid_imgs[i - 1].shape[:2])
        lpls = cv.subtract(pyramid_imgs[i - 1], expand)
        cv.imshow("laplace_up_" + str(i), lpls)


def sobel_demo(img):
    grad_x = cv.Sobel(img, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradx", gradx)
    cv.imshow("grady", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradxy", gradxy)

def laplacian_demo(img):
    # dst = cv.Laplacian(img, cv.CV_32F)
    # lpls = cv.convertScaleAbs(dst)

    # sobel_x_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dst = cv.filter2D(img, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)

    cv.imshow("laplacian", lpls)


def edge_demo(img):
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgray = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygray = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    
    # edge_output = cv.Canny(xgray, ygray, 50, 150)
    edge_output = cv.Canny(blurred, 50, 150)
    cv.imshow("Canny edge", edge_output)

    dst = cv.bitwise_and(img, img, mask=edge_output)
    cv.imshow("Color Edge", dst)


def line_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv.imshow("image-lines", img)

def line_detect_possible_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv.imshow("line detect possible", img)


def detect_circles_demo(img):
    dst = cv.pyrMeanShiftFiltering(img, 50, 120)
    cv.imshow("edge mohu", dst)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, maxRadius=0, minRadius=0)
    circles = np.uint16(np.round(circles))
    for i in circles[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)
    cv.imshow("hough circles", img)


def contours_demo(img):
    dst = cv.GaussianBlur(img, (13, 13), 0)
    # dst = cv.pyrMeanShiftFiltering(img, 20, 80)
    cv.imshow("blurred", dst)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("threshold", binary)

    binary = cv.Canny(dst, 50, 100)
    cv.imshow("canny", binary)

    contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        cv.drawContours(img, contours, i, (0, 0, 255), -1)
    cv.imshow("detect contours", img)


def measure_demo(img):
    dst = cv.GaussianBlur(img, (13, 13), 0)
    
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    binary = cv.Canny(dst, 50, 130)

    cv.imshow("canny", binary)
    contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxarea = 0
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        if w * h > maxarea :
            maxarea = w * h
            mx = x
            my = y 
            mw = w
            mh = h
            mm = cv.moments(contour)
            # print(type(mm), mm['m00'])
            cx = mm['m10'] / mm['m00']
            cy = mm['m01'] / mm['m00']
            print(area, w, h)
    cv.circle(img, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)
    cv.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    cv.imshow("measure", img)


def erode_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode", dst)


def dilate_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    dst = cv.dilate(binary, kernel)
    cv.imshow("dilate", dst)


def open_demo(img):
    img = cv.GaussianBlur(img, (3, 3), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open", dst)


def watershed_demo(img):
    print(img.shape)

    blurred = cv.pyrMeanShiftFiltering(img, 10, 100)

    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    nb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(nb, kernel, iterations=3)
    cv.imshow("mor-opt", sure_bg)

    dist = cv.distanceTransform(sure_bg, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance-opt", dist_output)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("surface", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers=markers)
    img[markers==-1] = [0, 0, 255]
    cv.imshow("result", img)


def face_detect_demo(img):
    face_detector = cv.CascadeClassifier("/home/way/Learn/opencv/python/data/cascade.xml")
    if face_detector is None:
        print("create classifier error")
        exit(-1)
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        frame = img
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow("result", frame)
        if cv.waitKey(10) == 27:
            break

img = cv.imread("/home/way/Learn/opencv/python/pos/sample.png")
if img is None:
    print("cv.imread()")
    exit(1)
# cv.imshow("img", img)
# video_demo()

t1 = cv.getTickCount()
extrace_object_demo()
t2 = cv.getTickCount()
print("time: %s ms"%((t2 - t1) / cv.getTickFrequency() * 1000))
cv.waitKey(0)

cv.destroyAllWindows()