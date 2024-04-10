import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools import show_image_list


def gsoc(frame):
    backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()
    img = backSub.apply(frame)
    return img


def dog(frame, kernel_0, sigma_0, kernel_1, sigma_1):
    gaussianBlur_0 = cv2.GaussianBlur(frame, (kernel_0, kernel_0), sigma_0)
    gaussianBlur_1 = cv2.GaussianBlur(frame, (kernel_1, kernel_1), sigma_1)
    return gaussianBlur_0 - gaussianBlur_1


def mog(self, frame):
    img = self.backGroundSubMOG2.apply(frame)
    return img


def knn(self, frame):
    img = self.backGroundSubKNN.apply(frame)
    return img


def get_frame(cap):
    ret, frame = cap.read()
    # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture("dataset/video/color/video.avi")

    bgs_mog2 = cv2.createBackgroundSubtractorMOG2()
    bgs_knn = cv2.createBackgroundSubtractorKNN()
    bgs_gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()
    # bgs_ = cv2.bgsegm.createBackgroundSubtractorGSOC()

    # ----------------------------------------
    ax0 = plt.subplot(2, 2, 1)
    plt.title("frame")
    ax1 = plt.subplot(2, 2, 2)
    plt.title("MOG")
    ax2 = plt.subplot(2, 2, 3)
    plt.title("KNN")
    ax3 = plt.subplot(2, 2, 4)
    plt.title("GSOC")

    # ----------------------------------------
    im0 = ax0.imshow(get_frame(cap))
    im1 = ax1.imshow(bgs_mog2.apply(get_frame(cap)))
    im2 = ax2.imshow(bgs_knn.apply(get_frame(cap)))
    im3 = ax3.imshow(bgs_gsoc.apply(get_frame(cap)))

    def update(i):
        im0.set_data(get_frame(cap))
        im1.set_data(bgs_mog2.apply(get_frame(cap)))
        im2.set_data(bgs_knn.apply(get_frame(cap)))
        im3.set_data(bgs_gsoc.apply(get_frame(cap)))

    ani = FuncAnimation(plt.gcf(), update, interval=1)
    plt.show()
