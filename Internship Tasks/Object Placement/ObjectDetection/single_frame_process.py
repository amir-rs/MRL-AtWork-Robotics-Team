from matplotlib import pyplot as plt
import time
import cv2
import numpy as np


# preprocess_frame:
#   TAKES [frame],
#   DOES [flip and grayscale],
#   RETURN [frame]
def preprocess_frame(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


# median_adaptive_th:
#   TAKES [frame],
#   DOES [1-l2)median blur. 2-l4)adaptive gaussian threshold on median blured frame.],
#   RETURN [frame: median_adaptive_th]
def median_adaptive_th(frame):
    # blur = cv2.fastNlMeansDenoising(frame, 7, 21)
    blur = cv2.medianBlur(frame, 13)
    # blur = cv2.GaussianBlur(frame, (7, 7), 13)
    median_adaptive_th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 7
    )
    return median_adaptive_th


# median_th:
#   TAKES [frame],
#   DOES [  1-l1)median blur.
#           2-l2)thresholds blured frame with BINARY_INV, OTSU methods],
#   RETURN [frame: median_th]
def median_th(frame):
    blur = cv2.medianBlur(frame, 11)
    median_th = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    return median_th


# get_contours:
#   TAKES [frame],
#   DOES [  1-l1)get the contours founded in the frame],
#   RETURN [frame]
def get_contours(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# draw_rectangle:
#   TAKES [frame, contours],
#   DOES [  1-l1_l8) set first contour in the set as initial value for minimum contour.
#           2-l9_l16) go through a for loop and check if there was any smaller contour,
#                     replace it as minimum and save it's dimentions as x, y, w, h.
#           3-l18) it draws all contours, but the minimum contour has bolder margin.]
#   RETURN [frame]
def draw_rectangle(frame, contours):
    error_persent = 1.30
    (xm, ym, wm, hm) = cv2.boundingRect(contours[0])
    wm = int(wm * error_persent)
    hm = int(hm * error_persent)
    # print(f"area:{cv2.contourArea(contours[0])}, x:{x} y:{y}, h:{h} w:{w}")
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,100,100), 2)

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


# create_mask:
#   TAKES [mask, contour]
#   DOES [  1-l1_l7) set first contour in the set as initial value for minimum contour.
#           2-l9_l29) go through a for loop and check if there was any smaller contour,
#                     replace it as minimum and save it's dimentions as w, h.]
#   RETURNS [mask (which has drawn contours on it), min_contour: [h:int, w:int]]
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
        # print(f"contour_area:{contour_area} threshold:{(mask.shape[0]*mask.shape[1])*0.002}")

        #   Filtering based on size
        # if contour_area > (mask.shape[0]*mask.shape[1])*0.002:

        #   GET MINIMUM AREA RECANGLE
        # rect =  cv2.minAreaRect(cont)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(mask, [box], 0, -1, -1)

        (x, y, w, h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        if contour_area < min_contour_area:
            min_contour = [h, w]
            # TESTING PORPUSES
            # minCont = cont
    # TESTING PORPUSES
    # (x,y,w,h) = cv2.boundingRect(minCont)
    # w = int(w * error_persent)
    # h = int(h * error_persent)
    # cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)

    return [mask, min_contour]


# convert_cm2pixle:
#   TAKES [ dimention(dimention of object in cm:[width, height]) ],
#           lens( camera lens specifications:[focal_length, pixel_pitch, lens_distance])]
#   DOES [  1-l1_l2) read dimention values from dimention variable to seperate variables
#           2-l3_l8) calculate width and height in pixels from this formula:
#                       pixels = real_distance * lens['focal_length'] / (lens['pixel_pitch'] * lens['lens_distance']) ]
#   RETURNS [   dict:[width: width on pixels, height: height in pixels]   ]
def convert_cm2pixle(dimention, lens):
    width = dimention["width"]
    height = dimention["height"]
    p_height = (
        height * lens["focal_length"] / (lens["pixel_pitch"] * lens["lens_distance"])
    )
    p_width = (
        width * lens["focal_length"] / (lens["pixel_pitch"] * lens["lens_distance"])
    )
    # pixels = real_distance * lens['focal_length'] / (lens['pixel_pitch'] * lens['lens_distance'])
    return {"width": p_width, "height": p_height}


# create_template:
#   TAKES [ dict: dimention[height, width]]
#   DOES [  1-l1) create numpy array with the given dimention ]
#   RETURNS [ created numpy array ]
def create_template(dimention):
    return np.ones((dimention["height"], dimention["width"]), dtype="uint8") * 255


# grid:
#   TAKES [ frame: frame
#           list: grid_cell[height, width]( this is the measure, that's been extracted from the smallest obstacle contour, founded in POV frame
#                                           and is used here as step value for our 'for' loop to create grid by this size on the main frame)
#           list: object[height, width]( this is the object which we are trying to choose a location for.) ]
#   DOES [ 1-l1_l6) read all dimention and values from given lists and frame to seperate varialbes.
#          2-l20-l32) loop through pixles of the frame by the step size of cell dimentions and check all pixels of that particular area,
#                   by slicing the frame with [loop itteration value (h and w) + object dimentions (object_height and object_width)] and call (all()) method.
#                     * step through each column of a row, then step row by row
#                     ** those commented lines were used to create a miniature grid frame by considering divide and conquer concepts
#                        but it wasn't necessary    ]
#   RETURNS [   a list of available grid cells by having their (w, h, w+object_width, h+object_height) in two sets of tuples]
#
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

    # grid_w = int(frame_w / cell_width)
    # grid_h = int(frame_h / cell_height)
    # grid = np.ones((grid_h, grid_w), dtype="uint8") * 255

    # Represent cell in an image for illustration popuses.
    # cell = np.ones((cell_height, cell_width), dtype="uint8") * 255

    # print(f"frame H&W:   {(frame_h, frame_w )}\ncells H&W:   {(cell_height, cell_width)}\ngrid H&W:    {(grid_h,grid_w)}")

    for h in range(0, frame_h, cell_height):
        for w in range(0, frame_w, cell_width):
            # print(f"frame[{h}:{h+cell_height}, {w}:{w+cell_width}]")
            if frame[h : h + object_height, w : w + object_width].all():
                # grid[int(((h+cell_height)/cell_height)-2), int(((w+cell_width)/cell_width)-2)] = 0
                # grid[int(((h)/cell_height)-1), int(((w)/cell_width)-1)] = 0
                # print(f"h:{h},w:{w},h+o_h:{h+object_height},w+o_w:{w+object_width}")
                available.append([(w, h), (w + object_width, h + object_height)])
                # print(f"----{int((h/cell_height)-1), int((w/cell_width)-1)}")
            # else:
            #    #grid[int(((h+cell_height)/cell_height)-2), int(((w+cell_width)/cell_width)-2)] = 255
            #    #grid[int(((h)/cell_height)-1), int(((w)/cell_width)-1)] = 255
            #    #print(f"----{int((h/cell_height)-1), int((w/cell_width)-1)}")
    return available


# star_map:
# DOES [ draw a map of available pixels by showing them with stars. ]
# just for illustration purposes
def star_map(grid_frame):
    cv2.imshow("grid_frame", grid_frame)
    for x1 in grid_frame:
        print()
        for x2 in x1:
            if x2 == 255:
                print("* ", end="")
            else:
                print("  ", end="")


# draw_available_grids:
#   TAKES [ frame, list: available_grids[ tuple: tuple0, tuple: tuple1]]
#   DOES [  draw all availabe grids on the give frame.  ]
#   RETURNS [ nothing. ]
def draw_available_grids(frame, available_grids):
    # print(available_grids)
    for grid in available_grids:
        # cv2.rectangle(frame, (w, h), (w+object_width, h+object_height), (255,0,255), 1)
        cv2.rectangle(frame, grid[0], grid[1], (255, 255, 0), 1)


# Implementation with example frames -----------------------------------------
address_list = [
    "./img samples/01.jpg",
    "./img samples/02.jpg",
    "./img samples/03.jpg",
    "./img samples/04.jpg",
    "./img samples/05.jpg",
]
# lens = {
#    'focal_length'  : 4.7 mili,
#    'pixel_pitch'   : 2.2 micro,
#    'lens_distance' : 0,
# }

obj_dim = [(80, 150), (10, 5), (8, 8)]
# temporary set pixle dimentions as static
p_dim = {"width": 80, "height": 400}

while True:
    # ---------- BEGINING TO READ
    frame = cv2.imread(address_list[2])
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    # frame = cv2.resize(frame, (1280, 720))
    # -- -- -- BEGINING TO DO COMPUTING ON FRAMES
    # -- -- -- create a copy from original frame  for illustration purposes
    original_frame = cv2.flip(frame.copy(), 1)

    # -- -- -- preprocess frames (fliping, bgr2gray, resize)
    frame = preprocess_frame(frame)

    # -- -- -- blur the frame and get a threshold
    threshold = median_th(frame)
    # threshold = median_adaptive_th(frame)

    # -- -- -- extract contours from the thresholded frame
    cntrs = get_contours(threshold)

    # -- -- -- draw contours on the original frame
    original_frame = draw_rectangle(original_frame, cntrs)

    # -- -- -- loop through contours and create a mask of True(s) and Flase(s)
    # print(frame.shape[:2])
    mask = np.ones(frame.shape[:2], dtype="uint8") * 255

    # -- -- -- draw contours on 'mask' and save the smallest obstacle dimention as 'grid_cell'
    mask, grid_cell = create_mask(mask, cntrs)
    # grid_cell_w = min_contour[0]
    # grid_cell_h = min_contour[1]
    # print(f"grid_cell_w:{grid_cell_w} grid_cell_h:{grid_cell_h}")
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # -- -- -- convert given size in cm to pixles
    # p_dim = convert_cm2pixle(obj_dim[0], lens)
    # draw_rectangle(original_frame, [p_dim])

    # -- -- -- get a list of availabe grid cells on the given frame(mask in this example)
    available_grids = grid(mask, grid_cell, obj_dim[0])

    # -- -- -- CHECK [grid_frame] GRID BY GRID FOR A PLACE WITH DIMENTION OF OBJECT THAT'S BEEN CHOOSED TO PLACE.
    # -- -- -- ATTENTION: THIS TASK IS DONE IN PREVIOUS PART BY {grid} FUNCTION.
    # available_grids = star_map(grid_frame)
    # print(f"available grid:{available_grids}")
    # original_frame = draw_rectangle(original_frame, cntrs)
    # cv2.rectangle(grid_frame, available_grids[0], available_grids[1], (0,0,255), -1)

    # -- -- -- DRAW AVAILABLE GRIDS ON ORIGINAL FRAME ALONG THE MASKED AREA AND...
    draw_available_grids(masked_frame, available_grids)

    # print("----------------------------------------------------------")
    # print("{img} shape: {shape}, dataType:{dtype}".format(img='mask', shape=mask.shape, dtype=mask.dtype))
    # print("{img} shape: {shape}, dataType:{dtype}".format(img='original_frame', shape=original_frame.shape, dtype=original_frame.dtype))
    # print("{img} shape: {shape}, dataType:{dtype}".format(img='masked_frame', shape=masked_frame.shape, dtype=masked_frame.dtype))
    # cv2.imshow("mask", mask)
    # cv2.imshow("Original", original_frame)
    cv2.imshow("masked_frame", masked_frame)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
