def empty(tst):
        pass

def trackbar(minval, maxval, frameSize=[520, 170]):
    frameSize[0] = 0
    frameSize[1] = 0
    frame = cv2.namedWindow("FrameSetup")
    cv2.resizeWindow("FrameSetup", frameSize[0], frameSize[1])
    cv2.createTrackbar("HUEmin", "FrameSetup", minval['HUE'][0], minval['HUE'][1], empty)
    cv2.createTrackbar("HUEmax", "FrameSetup", maxval['HUE'][0], maxval['HUE'][1], empty)
    cv2.createTrackbar("SATmin", "FrameSetup", minval['SAT'][0], minval['SAT'][1], empty)
    cv2.createTrackbar("SATmax", "FrameSetup", maxval['SAT'][0], maxval['SAT'][1], empty)
    cv2.createTrackbar("VALmin", "FrameSetup", minval['VAL'][0], minval['VAL'][1], empty)
    cv2.createTrackbar("VALmax", "FrameSetup", maxval['VAL'][0], maxval['VAL'][1], empty)

# Create Trackbars -----------
minval = {"HUE":(0, 179),
          "SAT":(0, 179),
          "VAL":(0, 179)}
maxval = {"HUE":(0, 179),
          "SAT":(0, 179),
          "VAL":(0, 179)}
trackbar(minval, maxval)

# Read frames and validate reading (Beginning of while)----------
    #HUEmin = cv2.getTrackbarPos("HUEmin", "FrameSetup")
    #HUEmax = cv2.getTrackbarPos("HUEmax", "FrameSetup")
    #SATmin = cv2.getTrackbarPos("SATmin", "FrameSetup")
    #SATmax = cv2.getTrackbarPos("SATmax", "FrameSetup")
    #VALmin = cv2.getTrackbarPos("VALmin", "FrameSetup")
    #VALmax = cv2.getTrackbarPos("VALmax", "FrameSetup")
    #valMin = np.array([HUEmin, SATmin, VALmin])
    #valMax = np.array([HUEmax, SATmax, VALmax])

# ------------------------------- mask stuff
# convert to HSV
frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Implement values on the masked frame -----------------
    #masked = cv2.inRange(frameHSV, valMin, valMax)


# ------------------------------- plotlib stuff
#images = [frameBlured, frameThresh, frame]
    #titles = ['frameBlured', 'frameThresh', 'frame']
    #for i in range(3):
    #    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #    plt.title(titles[i])
    #    plt.xticks([]),plt.yticks([])
    #plt.show()
