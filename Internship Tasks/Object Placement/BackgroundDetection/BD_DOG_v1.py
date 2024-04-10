import cv2 as cv


def dog(frame, low_kernel, low_sigma, high_kernel, high_sigma):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    low_gaussianBlur = cv.GaussianBlur(frame, (low_kernel, low_kernel), low_sigma)
    high_gaussianBlur = cv.GaussianBlur(frame, (high_kernel, high_kernel), high_sigma)
    return low_gaussianBlur - high_gaussianBlur


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
    kernel_CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    kernel_RECT = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # read frame
    cap = cv.VideoCapture(address)
    success, frame = cap.read()

    while success:
        # Process frame
        processed_frame = cv.morphologyEx(
            frame, cv.MORPH_OPEN, kernel_RECT, iterations=1
        )
        cv.imshow("processed_frame", processed_frame)

        blured_frame = cv.GaussianBlur(processed_frame, (11, 11), 3)
        # cv.imshow("blured_frame", blured_frame)

        foreground_mask = dog(
            blured_frame, low_kernel=3, low_sigma=0, high_kernel=5, high_sigma=0
        )
        cv.imshow("foreground_mask", foreground_mask)

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
