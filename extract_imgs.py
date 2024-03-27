import os
import sys
import cv2


def extract(ROOT):
    if not os.path.isdir("./" + ROOT):
        os.mkdir("./" + ROOT)
    cap = cv2.VideoCapture("./" + ROOT + ".mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # frame = frame[180:801, 800:1581]
            cv2.imwrite("./" + ROOT + "/" + str(i) + ".jpg", frame)
            print(i)
        i += 1

    cap.release()


ROOT = str(sys.argv[1])
extract(ROOT)
