import cv2

frame = cv2.imread("../dataset2/dac/M20_100/M20_100_1.jpg")

while True:
    cv2.imshow("M20_100_1.jpg", frame)
    if cv2.waitKey(0) == ord("q"):
        break
