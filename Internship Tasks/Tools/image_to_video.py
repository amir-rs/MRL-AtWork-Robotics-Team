import cv2
import numpy as np
import os
from os.path import isfile, join


def convert_frames_to_video(pathIn, static_name_length, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[static_name_length:-4]))
    for i in range(len(files)):
        filename = pathIn + "/" + files[i]
        print(filename)
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    obj_list = [
        {"name": "Axis", "length": 5},
        {"name": "Bearing", "length": 8},
        {"name": "Bearing_Box", "length": 12},
        {"name": "Distance_tube", "length": 14},
        {"name": "F20_20_B", "length": 9},
        {"name": "F20_20_G", "length": 9},
        {"name": "M20", "length": 4},
        {"name": "M20_100", "length": 8},
        {"name": "Motor", "length": 6},
        {"name": "R20", "length": 4},
    ]
    for obj in obj_list:
        pathIn = "../dataset/dataset2/dac/" + str(obj["name"])

        pathOut = str(obj["name"]) + ".avi"
        fps = 45.0
        convert_frames_to_video(pathIn, obj["length"], pathOut, fps)


if __name__ == "__main__":
    main()
