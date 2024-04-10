import cv2
import time
import numpy as np

# Version 1.10 --------------------------------------------------------
# ATTENTION: modes are designed to represent 3 states of detection method in one code
# MODE 0: threshold frames based on OTSU method.
# MODE 1: threshold frames dynamicaly based on THRESH_BINARY_INV method and threshold value is taken from trackbar.
# MODE 2: THERE IS NO BINARY THRESHOLD! it uses color masking and takes values(SAT, HEU, VAL) from trackbar to filter and designates the asked color.
# --------------------------------------------------------


def empty(tst):
        pass

def trackbar(modes = [0, 1, 2, 3], frameSize=[520, 170]):
    minval = {  "HUE":(0, 179),
                "SAT":(0, 179),
                "VAL":(0, 179)}
    maxval = {  "HUE":(0, 179),
                "SAT":(0, 179),
                "VAL":(0, 179)}
    
    frame = cv2.namedWindow("Detection Tweaks")
    cv2.resizeWindow("Detection Tweaks", frameSize[0], frameSize[1])
    
    cv2.createTrackbar("mode", "Detection Tweaks", modes[0], modes[2], empty)
    cv2.createTrackbar("thresh", "Detection Tweaks", 0, 255, empty)
    cv2.createTrackbar("HUEmin", "Detection Tweaks", minval['HUE'][0], minval['HUE'][1], empty)
    cv2.createTrackbar("HUEmax", "Detection Tweaks", maxval['HUE'][0], maxval['HUE'][1], empty)
    cv2.createTrackbar("SATmin", "Detection Tweaks", minval['SAT'][0], minval['SAT'][1], empty)
    cv2.createTrackbar("SATmax", "Detection Tweaks", maxval['SAT'][0], maxval['SAT'][1], empty)
    cv2.createTrackbar("VALmin", "Detection Tweaks", minval['VAL'][0], minval['VAL'][1], empty)
    cv2.createTrackbar("VALmax", "Detection Tweaks", maxval['VAL'][0], maxval['VAL'][1], empty)
    

trackbar()


try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 720)    
except:
    print("Can't read video frame.")

while True:
    mode = cv2.getTrackbarPos("mode", "Detection Tweaks")
    threshValue = cv2.getTrackbarPos("thresh", "Detection Tweaks")
    
    
    
    isReadOk, frame = cap.read()
    frame = cv2.flip(frame, 1)
    begin = time.time()
    
    # -----------------------------
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBlured = cv2.GaussianBlur(frameGray, (3, 3), 0)
    
    if mode == 0:
        frameThresh = cv2.threshold(frameBlured, 100, 255,
                                cv2.THRESH_OTSU)[1]
        
    elif mode == 1:
        frameThresh = cv2.threshold(frameBlured, threshValue, 255,
                                cv2.THRESH_BINARY_INV)[1]
        frameThresh = ~frameThresh
        
    elif mode == 2:
        HUEmin = cv2.getTrackbarPos("HUEmin", "Detection Tweaks")
        HUEmax = cv2.getTrackbarPos("HUEmax", "Detection Tweaks")
        SATmin = cv2.getTrackbarPos("SATmin", "Detection Tweaks")
        SATmax = cv2.getTrackbarPos("SATmax", "Detection Tweaks")
        VALmin = cv2.getTrackbarPos("VALmin", "Detection Tweaks")
        VALmax = cv2.getTrackbarPos("VALmax", "Detection Tweaks")
        lowerBound = np.array([HUEmin, SATmin, VALmin])
        upperBound = np.array([HUEmax, SATmax, VALmax])
        frameThresh = cv2.inRange(frame, lowerBound, upperBound)
    
    
    contours, hierarchy = cv2.findContours(frameThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        cv2.drawContours(frame, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10,0,10), 2)
    # -----------------------------
    
    end = time.time()
    fps = 1 /(end-begin)
    cv2.putText(frame, f"fps:{int(fps)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)
    
    cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break