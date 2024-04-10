import cv2
class Trackbar:
    def __init__(self, minval, maxval):
        cv2.namedWindow("FrameSetup")
        cv2.resizeWindow("FrameSetup", 520, 170)
        cv2.createTrackbar("HUEmin", "FrameSetup", minval{'HUE'}, self.empty)
        cv2.createTrackbar("HUEmax", "FrameSetup", maxval{'HUE'}, self.empty)
        cv2.createTrackbar("SATmin", "FrameSetup", minval{'SAT'}, self.empty)
        cv2.createTrackbar("SATmax", "FrameSetup", maxval{'SAT'}, self.empty)
        cv2.createTrackbar("VALmin", "FrameSetup", minval{'VAL'}, self.empty)
        cv2.createTrackbar("VALmax", "FrameSetup", maxval{'VAL'}, self.empty)
        
    def empty(self, tst):
            return tst
