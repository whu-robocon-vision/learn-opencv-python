import pyrealsense2 as rs
import cv2 as cv
import sys
import numpy as np

def nothing(x):
    pass

def extrace_object_demo():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    # capture = cv.VideoCapture(0)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    cv.namedWindow("frame")
    cv.createTrackbar('R', "frame", 81, 255, nothing)
    cv.createTrackbar('G', 'frame', 53, 255, nothing)
    cv.createTrackbar('B', 'frame', 84, 255, nothing)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue
        color_frame = np.asanyarray(color_frame.get_data())

        frame = cv.GaussianBlur(color_frame, (7, 7), 0)
        # cv.imshow("blurred", frame)
        mask = cv.inRange(frame, lowerb=np.array([0, 0, cv.getTrackbarPos('R', 'frame')]), upperb=np.array([cv.getTrackbarPos('B', 'frame'), cv.getTrackbarPos('G', 'frame'), 255]))

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)

        contours, heriachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):

            epsilon = 0.1 * cv.arcLength(contour, True)
            contour = cv.approxPolyDP(contour, epsilon, True)

            # contour = cv.convexHull(contour)

            contour_area = cv.contourArea(contour)
            rect = cv.minAreaRect(contour)
            w, h = rect[1]
            cv.drawContours(mask, contours, i, (0, 0, 255), 3)
            if  contour_area > 0.8 * w * h and contour_area > 1000:
                # cv.drawContours(frame, contours, i, (255, 0, 255), 2)
                
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(frame, [box], 0, (0, 0, 255), 3)
                mm = cv.moments(contour)
                cx = mm['m10'] / mm['m00']
                cy = mm['m01'] / mm['m00']
                cv.circle(frame, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)
                
        cv.imshow("mask", mask)
        cv.imshow("frame", frame)

        c = cv.waitKey(1)
        if c == 27:
            cv.destroyAllWindows()
            break


#     while(True):
#         frames = pipeline.wait_for_frames()
#         frame = frames.get_color_frame()


#         # ret, frame = capture.read()
#         # if ret == False:
#         #     break

#         # lower_hsv = np.array([156, 60, 46])
#         # upper_hsv = np.array([180, 255, 255])
#         # frame = cv.pyrMeanShiftFiltering(frame, 20, 100)
#         frame = cv.GaussianBlur(frame, (3, 3), 0)
#         # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         # mask1 = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
#         mask = cv.inRange(frame, lowerb=np.array([0, 0, 110]), upperb=np.array([110, 110, 255]))
#         kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#         mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
#         # mask = cv.bitwise_or(mask1, mask2)

        
#         # dst = cv.bitwise_and(frame, frame, mask=mask)

#         contours, heriachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         for i, contour in enumerate(contours):
#             contour_area = cv.contourArea(contour)
#             x, y, w, h = cv.boundingRect(contour)
#             cv.drawContours(mask, contours, i, (0, 0, 255), 3)
#             if contour_area > 0.8 * w * h:
#                 cv.drawContours(frame, contours, i, (255, 0, 255), 2)

#         cv.imshow("mask", mask)
#         cv.imshow("video", frame)
        
#         if cv.waitKey(40) == 27:
#             break

t1 = cv.getTickCount()
extrace_object_demo()
t2 = cv.getTickCount()
print("time: %s ms"%((t2 - t1) / cv.getTickFrequency() * 1000))

# cv.destroyAllWindows()