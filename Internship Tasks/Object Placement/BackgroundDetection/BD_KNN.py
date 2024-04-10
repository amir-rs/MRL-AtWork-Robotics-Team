import numpy as np
import cv2 as cv


# frame = cv.imread("../test_images/Axis_4.jpg")


def draw_rect_for_contours(frame, contours, hierarchy):
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
        if w > 80 and h > 80:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)

    return frame


if __name__ == "__main__":
    address = (
        "/home/zakaria/Documents/Projects/AtworkTasks/dataset/video/color/video.avi"
    )
    # read frame
    cap = cv.VideoCapture(address)
    success, frame = cap.read()

    fgbg = cv.createBackgroundSubtractorKNN()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    while success:
        foreground_mask = fgbg.apply(frame)
        cv.imshow("foreground_mask", foreground_mask)

        foreground_mask_processed = cv.morphologyEx(
            foreground_mask, cv.MORPH_GRADIENT, kernel
        )
        cv.imshow("foreground_mask_processed", foreground_mask_processed)

        # detect obj and draw contour

        contours, hierarchy = cv.findContours(
            foreground_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        draw_rect_for_contours(
            frame,
            contours,
            hierarchy,
        )

        cv.imshow("frame", frame)

        success, frame = cap.read()
        key = cv.waitKey()
        if key == ord("q"):
            break
        if key == ord("p"):
            cv.waitKey(-1)

    cv.destroyAllWindows()
