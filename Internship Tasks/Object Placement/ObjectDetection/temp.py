# First import library
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse
import os.path
import time


# Create object for parsing command-line options
parser = argparse.ArgumentParser(
    description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded."
)

# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")

# Parse the command line arguments to an object
args = parser.parse_args()

# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

kernel_MORPH_RECT = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
kernel_MORPH_CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
kernel_MORPH_ELLIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


def process(
    frame,
    erode_iter=3,
    dilate_iter=4,
    morph_kernel=kernel_MORPH_CROSS,
    gaussian_kernel_size=13,
):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blured_frame = cv.GaussianBlur(
        frame, (gaussian_kernel_size, gaussian_kernel_size), 0
    )
    # blur = cv.medianBlur(frame, 7)

    ret, threshold_frame = cv.threshold(
        blured_frame, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
    )

    """ threshold_frame = cv.adaptiveThreshold(
        blured_frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 20
    ) """

    threshold_frame = cv.erode(threshold_frame, morph_kernel, iterations=erode_iter)
    out_frame = cv.dilate(threshold_frame, morph_kernel, iterations=dilate_iter)

    # out_frame = cv.morphologyEx(threshold_frame, cv.MORPH_OPEN, kernel_RECT)

    return out_frame


def auto_canny(frame, sigma=0.33):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_frame, (11, 11), 0)

    # compute the median of the single channel pixel intensities
    v = np.median(blurred)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(blurred, lower, upper)

    # return the edged image
    return edged


def detect_contour(frame):
    contours, hierarchy = cv.findContours(
        frame,
        # cv.RETR_EXTERNAL,
        # cv.RETR_LIST,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    cv.drawContours(color_image, contours, -1, (255, 22, 22), 4)

    return contours


def draw_objects(frame, contours):
    # computes the bounding box for the contour, and draws it on the frame,
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) < 8 and cv.contourArea(contour) > 250:
            M = cv.moments(contour)
            x_center = int((M["m10"] / M["m00"]))
            y_center = int((M["m01"] / M["m00"]))

            # cv.drawContours(frame, [contour], 0, (0, 0, 0), 6)
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 4)

        # {"x_center": x_center, "y_center": y_center}
    return frame


if __name__ == "__main__":
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        config.enable_device_from_file(args.input)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                # print("Depth camera with Color sensor")
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        # Start streaming from file
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # /////////////////////////////////  Processing configurations /////////////////////////////////
        # ---------- decimation ----------
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)

        # ---------- spatial ----------
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        spatial.set_option(rs.option.holes_fill, 3)

        # ---------- hole_filling ----------
        hole_filling = rs.hole_filling_filter()

        # ---------- disparity ----------
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        # /////////////////////////////////  Colorizer configurations /////////////////////////////////
        colorizer = rs.colorizer()
        colorizer.set_option(
            rs.option.visual_preset, 0
        )  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        colorizer.set_option(rs.option.color_scheme, 3)
        colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
        colorizer.set_option(rs.option.min_distance, 0.0)
        colorizer.set_option(rs.option.max_distance, 0.5)

        # /////////////////////////////////  Allignment configurations /////////////////////////////////
        align = rs.align(rs.stream.color)

    finally:
        pass

    # Streaming loop
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        begin = time.time()

        # /////////////////////////////////  Get RGB frame /////////////////////////////////
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_colormap_dim = color_image.shape

        # /////////////////////////////////  Get DEPTH frame /////////////////////////////////
        depth_frame = aligned_frames.get_depth_frame()

        # Apply filters
        depth_frame = decimation.process(depth_frame)
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        # depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_colormap_dim = depth_image.shape

        ## If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #    resized_color_image = cv.resize(
        #        color_image,
        #        dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
        #        interpolation=cv.INTER_AREA,
        #    )
        #    images = np.hstack((resized_color_image, depth_image))
        # else:
        #    images = np.hstack((color_image, depth_image))

        # /////////////////////////////////  Processing frames /////////////////////////////////
        processed_frame = process(color_image)

        # /////////////////////////////////  Find contours and Draw /////////////////////////////////
        contours = detect_contour(processed_frame)
        color_frame = draw_objects(color_image, contours)
        depth_frame = draw_objects(depth_image, contours)

        end = time.time()
        fps = 1 / (end - begin)

        cv.putText(
            color_frame,
            f"fps:{int(fps)}",
            (5, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 250, 0),
            2,
        )

        cv.imshow("RGB Image", color_image)
        cv.imshow("DEPTH Image", depth_image)

        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            break
        if key == ord("p"):
            cv.waitKey(-1)
