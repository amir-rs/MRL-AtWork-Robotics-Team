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

kernel_MORPH_RECT = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
kernel_MORPH_CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
kernel_MORPH_ELLIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


def process(
    frame,
    erode_iter=3,
    dilate_iter=4,
    morph_kernel=kernel_MORPH_ELLIPSE,
    gaussian_kernel_size=13,
):
    if len(frame.shape) == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # frame = cv.GaussianBlur(frame, (gaussian_kernel_size, gaussian_kernel_size), 0)
    # frame = cv.bilateralFilter(frame, 13, 35, 35)

    ret, frame = cv.threshold(
        frame, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
    )

    frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, morph_kernel, iterations=1)
    frame = cv.morphologyEx(frame, cv.MORPH_OPEN, morph_kernel, iterations=2)

    # apply automatic Canny edge detection using the computed median
    v = np.median(frame)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    frame = cv.Canny(frame, lower, upper)
    frame = cv.dilate(frame, morph_kernel, iterations=1)

    return frame


def detect_contour(frame, return_option):
    contours_list = []
    contours, hierarchy = cv.findContours(
        frame,
        cv.RETR_EXTERNAL,
        # cv.RETR_LIST,
        # cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    scale = 0.85

    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.006 * peri, True)
        area = cv.contourArea(contour)
        if (
            area > 200
            and area < (scale * (frame.shape[0] * frame.shape[1]))
            and len(approx) < 20
        ):
            if return_option == "approx":
                contours_list.append(approx)
            elif return_option == "contour":
                contours_list.append(contour)

    # cv.drawContours(mask, contours, -1, (255), 4)
    return contours_list


def draw_contours(frame, contours, mode="bbox"):
    if mode == "bbox":
        # computes the bounding box for the contour, and draws it on the frame,
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 4)
    elif mode == "contour":
        cv.drawContours(frame, contours, -1, (0, 200, 0), 4)
        # frame = cv.dilate(frame, kernel_MORPH_RECT, iterations=2)


def post_process_depth(depth_frame):
    # depth_frame = decimation.process(depth_frame)
    depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    # depth_frame = temporal.process(depth_frame)
    depth_frame = disparity_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    return depth_frame


def contour_depth_value(depth_frame, contour, demo_frame=None, scale=0.3):
    x, y, w, h = cv.boundingRect(contour)
    scaled_x = int(x + scale * w)
    scaled_y = int(y + scale * h)
    scaled_w = int((1 - scale) * w)
    scaled_h = int((1 - scale) * h)

    depth_value = int(
        depth_frame[
            scaled_y : scaled_y + scaled_h,
            scaled_x : scaled_x + scaled_w,
        ].mean()
    )
    if demo_frame is not None:
        cv.rectangle(
            demo_frame,
            (scaled_x, scaled_y),
            (scaled_x + scaled_w, scaled_y + scaled_h),
            (0, 200, 0),
            4,
        )

        cv.putText(
            demo_frame,
            str(depth_value),
            (int(x + w / 2), int(y + h / 2)),
            cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
            1,
            0,
            3,
        )
        return depth_value

    return depth_value


def filter_contours(depth_frame, rgb_contours):
    # df_copy = depth_frame.copy()
    approved_contours = []

    frame_depth_val = int(np.mean(depth_frame))

    for rgb_contour in rgb_contours:
        rgb_cnt_depth_val = contour_depth_value(depth_frame, rgb_contour)
        if rgb_cnt_depth_val > frame_depth_val:
            approved_contours.append(rgb_contour)

    # cv.putText(
    #    df_copy,
    #    f"avg val frame:{frame_depth_val}",
    #    (5, 30),
    #    cv.FONT_HERSHEY_SIMPLEX,
    #    1,
    #    (0, 250, 0),
    #    2,
    # )

    # cv.imshow("df copy", df_copy)
    return approved_contours


def create_mask(frame_shape, contours):
    mask = np.ones(frame_shape[:2], dtype="uint8") * 0
    cv.drawContours(mask, contours, -1, (255), cv.FILLED)
    mask = cv.dilate(mask, kernel_MORPH_RECT, iterations=2)
    return mask


def process_contours(contours, kernel=kernel_MORPH_RECT):
    frame = np.ones(color_image.shape[:2], dtype="uint8") * 0
    cv.drawContours(frame, contours, -1, (255), 3)

    # frame = cv.bilateralFilter(frame, 13, 35, 35)
    frame = cv.dilate(frame, kernel, iterations=1)
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

    return detect_contour(frame, "contour")


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
        align_to = rs.stream.color
        align = rs.align(align_to)

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
        depth_frame = post_process_depth(
            depth_frame
        )  # Apply filters for post-processing
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_image = cv.cvtColor(depth_image, cv.COLOR_BGR2GRAY)
        depth_colormap_dim = depth_image.shape

        # /////////////////////////////////  Processing frames /////////////////////////////////
        processed_frame = process(color_image)

        # /////////////////////////////////  Find contours and Draw /////////////////////////////////
        contours = detect_contour(processed_frame, "approx")
        # temp = auto_canny(processed_frame)
        # contours = process_contours(contours)

        # /////////////////////////////////  Filter contours and Draw /////////////////////////////////
        contours = filter_contours(depth_image, contours)
        draw_contours(color_image, contours, mode="contour")

        # /////////////////////////////////  Find corners /////////////////////////////////
        mask = create_mask(depth_image.shape[:2], contours)

        # depth_image = np.float32(depth_image)

        dst = cv.cornerHarris(mask, 3, 3, 0.04)
        # dst = cv.dilate(dst, None)
        color_image[dst > 0.002 * dst.max()] = [0, 0, 255]

        end = time.time()
        fps = 1 / (end - begin)

        cv.putText(
            color_image,
            f"fps:{int(fps)}",
            (5, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 250, 0),
            2,
        )

        # cv.imshow("temp", temp)
        # cv.imshow("Aligned DEPTH Image", aligned_depth_frame)
        # cv.imshow("DEPTH Image", depth_image)

        # cv.imshow("color_image", color_image)
        cv.imshow("processed_frame", processed_frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            break
        if key == ord("p"):
            cv.waitKey(-1)
