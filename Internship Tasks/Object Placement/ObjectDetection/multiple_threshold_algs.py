from matplotlib import pyplot as plt
import time
import cv2
import numpy as np
from matplotlib.animation import FuncAnimation


def median_adaptive_th(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.fastNlMeansDenoising(frame, 7, 21)
    blur = cv2.medianBlur(frame, 13)
    # blur = cv2.GaussianBlur(frame, (7, 7), 13)
    median_adaptive_th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 7
    )
    return median_adaptive_th


def median_th(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(frame, 11)
    median_th = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    return median_th


def get_contours(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_rectangle(frame, contours):
    error_persent = 1.30
    (xm, ym, wm, hm) = cv2.boundingRect(contours[0])
    wm = int(wm * error_persent)
    hm = int(hm * error_persent)

    min_contour_area = cv2.contourArea(contours[0])
    for cont in contours:
        contour_area = cv2.contourArea(cont)
        (x, y, w, h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        if contour_area < min_contour_area:
            (xm, ym, wm, hm) = (x, y, w, h)

    cv2.rectangle(frame, (xm, ym), (xm + wm, ym + hm), (0, 0, 0), 5)
    return frame


def create_mask(mask, contours):
    error_persent = 1.30
    (x, y, w, h) = cv2.boundingRect(contours[0])
    w = int(w * error_persent)
    h = int(h * error_persent)
    min_contour = [h, w]
    min_contour_area = cv2.contourArea(contours[0])
    # Set a +1.3 coefficient as predicted error

    for cont in contours:
        contour_area = cv2.contourArea(cont)
        (x, y, w, h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        if contour_area < min_contour_area:
            min_contour = [h, w]

    return [mask, min_contour]


def grid(frame, grid_cell, object):
    available = []
    frame_h, frame_w = frame.shape[:2]
    cell_height = grid_cell[0]
    cell_width = grid_cell[1]
    object_height = object[0]
    object_width = object[1]
    print(
        f"f_h:{frame_h} -- f_w:{frame_w} -- c_h:{cell_height} -- c_w:{cell_width} -- o_h:{object_height} -- o_w:{object_width}"
    )

    for h in range(0, frame_h, cell_height):
        for w in range(0, frame_w, cell_width):
            # print(f"frame[{h}:{h+cell_height}, {w}:{w+cell_width}]")
            if frame[h : h + object_height, w : w + object_width].all():
                available.append([(w, h), (w + object_width, h + object_height)])
    return available


def draw_available_grids(frame, available_grids):
    for grid in available_grids:
        cv2.rectangle(frame, grid[0], grid[1], (255, 255, 0), 1)


def apply_method(frame, threshold_func):
    threshold = threshold_func(frame)
    cntrs = get_contours(threshold)
    original_frame = draw_rectangle(frame, cntrs)
    return original_frame


def get_frame(cap):
    ret, frame = cap.read()
    # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# Implementation with example frames -----------------------------------------
dataset_color = [
    "dataset/video/color/Axis.avi",
    "dataset/video/color/Bearing_Box.avi",
    "dataset/video/color/Bearing.avi",
    "dataset/video/color/Distance_tube.avi",
    "dataset/video/color/M20_100.avi",
    "dataset/video/color/R20.avi",
    "dataset/video/color/video.avi",
]
dataset_dac = [
    "dataset/video/dac/Axis.avi",
    "dataset/video/dac/Bearing_Box.avi",
    "dataset/video/dac/Bearing.avi",
    "dataset/video/dac/Distance_tube.avi",
    "dataset/video/dac/M20_100.avi",
    "dataset/video/dac/R20.avi",
]
base_folder = "/home/zakaria/Documents/Projects/AtworkTasks/"
if __name__ == "__main__":
    cap = cv2.VideoCapture(base_folder + dataset_dac[-1])

    # --------------------------------------

    ax0 = plt.subplot(2, 2, 1)
    plt.title("frame")
    ax1 = plt.subplot(2, 2, 2)
    plt.title("median_threshold_frame")
    ax2 = plt.subplot(2, 2, 3)
    plt.title("adaptive_median_threshold_frame")
    # ax3 = plt.subplot(2, 2, 4)
    # plt.title("KNN")

    # ----------------------------------------
    im0 = ax0.imshow(get_frame(cap))
    im1 = ax1.imshow(apply_method(get_frame(cap), median_th))
    im2 = ax2.imshow(apply_method(get_frame(cap), median_adaptive_th))
    # im3 = ax3.imshow()

    def update(i):
        im0.set_data(get_frame(cap))
        im1.set_data(apply_method(get_frame(cap), median_th))
        im2.set_data(apply_method(get_frame(cap), median_adaptive_th))
        # im3.set_data(bgs_gsoc.apply(get_frame(cap)))

    ani = FuncAnimation(plt.gcf(), update, interval=1)
    plt.show()
