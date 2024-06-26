from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


model = EfficientNetB0(include_top=False, weights="imagenet")

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224
batch_size = 64
data_dir = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images"


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
)

class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


## Standardize the data

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

for image, label in train_ds.take(0):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        aug_img = img_augmentation(image)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(label))
        plt.axis("off")

num_classes = len(class_names)

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

# model.compile(
#    optimizer="adam",
#    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    metrics=["accuracy"],
# )


# !mkdir -p saved_model
# model.save('saved_model/my_model')


model = tf.keras.models.load_model(
    "/home/erfan/Documents/Projects/AtworkTasks/PPT/PPT_bbox_classifier/saved_model/my_model"
)
model.summary()

epochs = 10
# history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)

# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label="Training Accuracy")
# plt.plot(epochs_range, val_acc, label="Validation Accuracy")
# plt.legend(loc="lower right")
# plt.title("Training and Validation Accuracy")
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label="Training Loss")
# plt.plot(epochs_range, val_loss, label="Validation Loss")
# plt.legend(loc="upper right")
# plt.title("Training and Validation Loss")
# plt.show()


model.summary()
len(model.trainable_variables)

cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/F20_20_horizontal/1442-1.jpg"
cavity_url = "/home/erfan/Documents/Projects/AtworkTasks/dataset/PPT/cavity_images/M20_100_horizontal/M20_100_downward506.jpg"

image = tf.keras.preprocessing.image.load_img(cavity_url)
image = tf.image.resize(
    image,
    [224, 224],
    # method=ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None,
)

img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)


try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2 as cv
    import argparse
    import os.path
    import time
    from pathlib import Path

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
    # if os.path.splitext(args.input)[1] != ".bag":
    #    print("The given file is not of correct file format.")
    #    print("Only .bag files are accepted")
    #    exit()

    kernel_MORPH_RECT = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_MORPH_CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    kernel_MORPH_ELLIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def process(
        frame,
        erode_iter=3,
        dilate_iter=4,
        morph_kernel=kernel_MORPH_CROSS,
        gaussian_kernel_size=13,
    ):
        frame = cv.GaussianBlur(frame, (gaussian_kernel_size, gaussian_kernel_size), 0)
        # frame = cv.bilateralFilter(frame, 13, 35, 35)

        ret, frame = cv.threshold(
            frame, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
        )

        frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, morph_kernel, iterations=1)
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, morph_kernel, iterations=2)
        frame = cv.morphologyEx(frame, cv.MORPH_GRADIENT, morph_kernel, iterations=1)
        # frame = cv.dilate(frame, morph_kernel, iterations=1)

        # apply automatic Canny edge detection using the computed median
        # v = np.median(frame)
        # lower = int(max(0, (1.0 - 0.33) * v))
        # upper = int(min(255, (1.0 + 0.33) * v))
        # frame = cv.Canny(frame, lower, upper)

        return frame

    def extract_bbox(frame, depth_frame, bbox_dim, draw_canvas):
        bbox_list = []
        contours, hierarchy = cv.findContours(
            frame,
            cv.RETR_EXTERNAL,
            # cv.RETR_LIST,
            # cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE,
        )
        scale = 0.85

        frame_depth_val = int(np.mean(depth_frame))

        half_bbox_dim = bbox_dim // 2
        for contour in contours:
            # Check depth information of each contour
            rgb_cnt_depth_val = contour_depth_value(depth_frame, contour)
            if rgb_cnt_depth_val > frame_depth_val:
                peri = cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, 0.02 * peri, True)
                area = cv.contourArea(contour)
                if (
                    area > 200
                    and area < (scale * (frame.shape[0] * frame.shape[1]))
                    and len(approx) < 20
                ):
                    M = cv.moments(contour)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    x, y, w, h = (
                        cX - half_bbox_dim,
                        cY - half_bbox_dim,
                        bbox_dim,
                        bbox_dim,
                    )

                    if draw_canvas is not None:
                        cv.rectangle(
                            draw_canvas, (x, y), (x + w, y + h), (0, 200, 220), 4
                        )

                    if sum(1 for n in [x, y] if n > 0) == 2:
                        # print(f"[(x, y), (x + w, y + h)]: {[(x, y), (x + w, y + h)]}")
                        # print(f"condition: {sum(1 for n in [x, y] if n>0)}")

                        bbox_list.append([(x, y), (x + w, y + h)])
                    # cv.circle(draw_canvas, (cX, cY), 7, (255, 255, 255), -1)
                    # cv.putText(draw_canvas, "center", (cX - 20, cY - 20),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # cv.drawContours(rgb_image, contours, -1, (0, 0, 255), 2)
        return bbox_list
        # if len(bbox_list) !=0:

    def crop_bounding_boxes(frame, bounding_boxes, base_dir_address, frame_counter):
        # [(x, y), (x + w, y + h)]
        # computes the bounding box for the contour, and draws it on the frame,
        for i, bbox in enumerate(bounding_boxes):
            # croped_bbox = frame[y:y+h, x:x+w]
            croped_bbox = frame[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
            print(f"draw:{bbox} at {base_dir_address}{frame_counter}_{i}.jpg")
            cv.imwrite(
                os.path.join(base_dir_address, f"{frame_counter}-{i}.jpg"), croped_bbox
            )
            # if croped_bbox is not None :
            #    cv.imwrite(f'{base_dir_address}/bboxed_label_{i}.jpg', croped_bbox)

    def post_process_depth(depth_frame):
        # depth_frame =
        # .process(depth_frame)
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

        mask = np.ones(frame_shape[:2], dtype="uint8") * 0
        cv.drawContours(mask, contours, -1, (255), cv.FILLED)
        mask = cv.dilate(mask, kernel_MORPH_CROSS, iterations=2)

        return mask

    if __name__ == "__main__":
        try:
            pipeline = rs.pipeline()
            config = rs.config()

            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            if args.input != "cam":
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

            pipeline.start(config)

            # Start streaming from file
            # profile = pipeline.start(config)

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            # depth_sensor = profile.get_device().first_depth_sensor()
            # depth_scale = depth_sensor.get_depth_scale()
            # print("Depth Scale is: ", depth_scale)

            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            # clipping_distance_in_meters = 1  # 1 meter
            # clipping_distance = clipping_distance_in_meters / depth_scale

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

        BaseDir = os.path.dirname(os.path.abspath(__file__))
        dataset_storage_path = os.path.join(BaseDir, "ppt_storage")
        frame_counter = 1
        # Streaming loop
        while True:
            key = cv.waitKey(1)

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            begin = time.time()

            # /////////////////////////////////  Get RGB frame /////////////////////////////////
            color_frame = aligned_frames.get_color_frame()
            bgr_image = np.asanyarray(color_frame.get_data())
            grayed_bgr_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

            # /////////////////////////////////  Get DEPTH frame /////////////////////////////////
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = post_process_depth(
                depth_frame
            )  # Apply filters for post-processing
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_image = cv.cvtColor(depth_image, cv.COLOR_BGR2GRAY)
            depth_colormap_dim = depth_image.shape

            # /////////////////////////////////  Processing frames /////////////////////////////////
            processed_frame = process(grayed_bgr_image)

            # /////////////////////////////////  Find contours and Draw /////////////////////////////////
            # show boundig_boxes
            bounding_boxes = extract_bbox(processed_frame, depth_image, 400, bgr_image)

            for i, bbox in enumerate(bounding_boxes):
                # croped_bbox = frame[y:y+h, x:x+w]
                croped_bbox = bgr_image[
                    bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]
                ]

                # Classify:

                bbox_resized = cv.resize(croped_bbox, (224, 224))

                img_array = tf.keras.utils.img_to_array(bbox_resized)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                cv.putText(
                    bgr_image,
                    f"{class_names[np.argmax(score)]} , {(100 * np.max(score)):.2f}",
                    (bbox[0][0], bbox[0][1] + 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

                # print(
                #    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                #        class_names[np.argmax(score)], 100 * np.max(score)
                #    )
                # )

            # if key == ord("r"):
            #    crop_bounding_boxes(
            #        grayed_bgr_image, bounding_boxes, dataset_storage_path, frame_counter
            #    )
            #    frame_counter += 1

            # /////////////////////////////////  Find corners /////////////////////////////////

            end = time.time()
            fps = 1 / (end - begin)

            cv.putText(
                bgr_image,
                f"fps:{int(fps)}",
                (5, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 250, 0),
                2,
            )

            cv.imshow("rgb_image", bgr_image)

            if key == ord("q"):
                cv.destroyAllWindows()
                break
            if key == ord("p"):
                cv.waitKey(-1)

finally:
    pass
