from boundingBox_detector import *
import tensorflow as tf


def classify(bbox, model):
    bbox_resized = cv.resize(bbox, (224, 224))
    img_array = tf.keras.utils.img_to_array(bbox_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    print(img_array)

    # predictions = model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])

    # cv.putText(
    #    bgr_image,
    #    f"{class_names[np.argmax(score)]} , {(100 * np.max(score)):.2f}",
    #    (bbox[0][0], bbox[0][1] + 30),
    #    cv.FONT_HERSHEY_SIMPLEX,
    #    1,
    #    (0, 0, 0),
    #    1,
    # )

    # print(
    #    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
    #        class_names[np.argmax(score)], 100 * np.max(score)
    #    )
    # )


if __name__ == "__main__":
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

        # /////////////////////////////////  Find and extract bounding boxes /////////////////////////////////
        bounding_boxes = extract_bbox(processed_frame, depth_image, 400, bgr_image)

        # /////////////////////////////////  Classification /////////////////////////////////
        # for bbox in bounding_boxes:
        #    # croped_bbox = frame[y:y+h, x:x+w]
        #    croped_bbox = bgr_image[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
        #    classify(croped_bbox)

        # /////////////////////////////////  Find corners /////////////////////////////////

        end = time.time()
        fps = 1 / (end - begin)

        # cv.putText(
        #    bgr_image,
        #    f"fps:{int(fps)}",
        #    (5, 30),
        #    cv.FONT_HERSHEY_SIMPLEX,
        #    1,
        #    (0, 250, 0),
        #    2,
        # )

        cv.imshow("rgb_image", bgr_image)

        if key == ord("q"):
            cv.destroyAllWindows()
            break
        if key == ord("p"):
            cv.waitKey(-1)
