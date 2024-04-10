from matplotlib import pyplot as plt
import time
import cv2 as cv
import numpy as np


def process(frame):
    blur = cv.GaussianBlur(frame, (11, 11), 0)
    blur = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # blur = cv.medianBlur(frame, 7)

    ret, threshold_frame = cv.threshold(
        blur, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
    )

    """ threshold_frame = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 20
    ) """

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # threshold_frame = cv.dilate(threshold_frame, kernel, iterations=1)
    # out_frame = cv.erode(threshold_frame, kernel, iterations=1)

    out_frame = cv.morphologyEx(threshold_frame, cv.MORPH_GRADIENT, kernel)

    return out_frame


def detect_contour(frame):
    contours, hierarchy = cv.findContours(
        frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # return (contours, hierarchy)
    return (contours, hierarchy)


def draw_objects(frame, contours, hierarchy):
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    height, width = frame.shape[:2]
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 10 and h > 10:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)

    return frame

    """ cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv.drawContours(frame, conts, -1, (22, 22, 22), 2, cv.LINE_8, hierarchy, 0)

    return frame """


if __name__ == "__main__":
    address = (
        "/home/zakaria/Documents/Projects/AtworkTasks/dataset/video/color/video.avi"
    )
    # read frame
    cap = cv.VideoCapture(address)
    success, frame = cap.read()

    while success:
        # Process frame
        processed_frame = process(frame)

        # Display Processed frame
        cv.imshow("processed_frame", processed_frame)

        # detect contours
        contours, hierarchy = detect_contour(processed_frame)

        # detect objects
        # objects = detect(contours, frame)

        # draw rectangle around objects
        mask = draw_objects(frame, contours, hierarchy)
        cv.imshow("output_frame", mask)

        success, frame = cap.read()
        key = cv.waitKey()
        if key == ord("q"):
            break
        if key == ord("p"):
            cv.waitKey(-1)


cv.destroyAllWindows()
