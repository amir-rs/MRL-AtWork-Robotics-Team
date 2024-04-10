import cv2 as cv

# frame = cv.imread("../test_images/Axis_4.jpg")


if __name__ == "__main__":
    address = (
        "/home/zakaria/Documents/Projects/AtworkTasks/dataset/video/color/video.avi"
    )

    cap = cv.VideoCapture(address)
    fgbg = cv.createBackgroundSubtractorMOG2()
    fgbg.setDetectShadows(False)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    success, frame = cap.read()

    while success:
        fgmask = fgbg.apply(frame)

        cv.imshow("frame", fgmask)

        foreground_mask_processed = cv.morphologyEx(
            fgmask, cv.MORPH_OPEN, kernel, iterations=1
        )
        cv.imshow("foreground_mask_processed", foreground_mask_processed)

        success, frame = cap.read()

        key = cv.waitKey()
        if key == ord("q"):
            break
        elif key == ord("p"):
            cv.waitKey(-1)

cv.destroyAllWindows()
