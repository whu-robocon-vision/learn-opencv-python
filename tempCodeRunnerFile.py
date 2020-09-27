epsilon = 0.1 * cv.arcLength(contour, True)
            contour = cv.approxPolyDP(contour, epsilon, True)