# blur_mode = cv.getTrackbarPos("Blur Modes", "Detection Tweaks")
# if blur_mode == 0:
#    diameter, sigmaColor, sigmaSpace = params[1]
#    blured_frame = cv.bilateralFilter(
#        foreground_mask, diameter, sigmaColor, sigmaSpace
#    )
# elif blur_mode == 1:
#    blured_frame = cv.GaussianBlur(foreground_mask, (9, 9), 3)
# cv.imshow("blured_frame", blured_frame)
