import cv2
from matplotlib import pyplot as plt
from ../../Tools.tools import show_image_list

# def gsoc(self):
#    cv.bgse


class BackgroundDetector(object):
    def __init__(self):
        self.backGroundSubMOG2 = cv2.createBackgroundSubtractorMOG2()
        self.backGroundSubKNN = cv2.createBackgroundSubtractorKNN()

    def doG(self, frame, kernel_0, sigma_0, kernel_1, sigma_1):
        gaussianBlur_0 = cv2.GaussianBlur(frame, (kernel_0, kernel_0), sigma_0)
        gaussianBlur_1 = cv2.GaussianBlur(frame, (kernel_1, kernel_1), sigma_1)
        return gaussianBlur_0 - gaussianBlur_1

    def moG2(self, frame):
        img = self.backGroundSubMOG2.apply(frame)
        return img

    def kNN(self, frame):
        img = self.backGroundSubKNN.apply(frame)
        return img

    def gMG(self, frame):
        pass


if __name__ == "__main__":

    # GET FRAME []
    frame = cv2.imread("../test_images/Axis_5.jpg")
    # frame = cv2.VideoCapture(0)

    bgd = BackgroundDetector()
    images_list = []

    backgroundframeDOG = bgd.doG(frame, 7, 7, 17, 13)
    images_list.append(backgroundframeDOG)

    backgroundframeKNN = bgd.kNN(frame)
    images_list.append(backgroundframeKNN)

    backgroundframeMOG2 = bgd.moG2(frame)
    images_list.append(backgroundframeMOG2)

    show_image_list(
        images_list,
        ["DOG", "KNN", "MOG2"],
        figsize=(12, 8),
    )
