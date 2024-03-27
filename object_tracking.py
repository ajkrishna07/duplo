import os
import sys
import csv
import cv2
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(text.split(".")[0])]


IMAGE_FOLDER = sys.argv[1]
IMAGES = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".jpg")]
IMAGES.sort(key=natural_keys)
OUTPUT_CSV_FILE = IMAGE_FOLDER + ".csv"

tracker = cv2.TrackerCSRT_create()
tracker_name = str(tracker).split()[0][1:]
first_frame = cv2.imread(os.path.join(IMAGE_FOLDER, IMAGES[0]))
roi = cv2.selectROI(first_frame)
ret = tracker.init(first_frame, roi)
cv2.destroyAllWindows()

(x, y, w, h) = tuple(map(int, roi))
first_line = [IMAGES[0][: IMAGES[0].index(".jpg")]]
first_line.append(x)
first_line.append(y)
first_line.append(w)
first_line.append(h)
with open(OUTPUT_CSV_FILE, "a") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(first_line)


for INPUT_IMAGE in IMAGES[1:]:
    frame = cv2.imread(os.path.join(IMAGE_FOLDER, INPUT_IMAGE))
    line = [INPUT_IMAGE[: INPUT_IMAGE.index(".jpg")]]
    frame_copy = frame.copy()
    success, roi = tracker.update(frame)
    (x, y, w, h) = tuple(map(int, roi))
    if success:
        pts1 = (x, y)
        pts2 = (x + w, y + h)
        cv2.rectangle(frame, pts1, pts2, (255, 125, 25), 2)
    cv2.imshow(str(INPUT_IMAGE), frame)
    key = cv2.waitKey(300)
    if key == ord("p"):
        cv2.destroyAllWindows()
        tracker = cv2.TrackerCSRT_create()
        roi = cv2.selectROI(frame_copy)
        ret = tracker.init(frame_copy, roi)
        (x, y, w, h) = tuple(map(int, roi))
    if key == ord("q"):
        break
    line.append(x)
    line.append(y)
    line.append(w)
    line.append(h)
    with open(OUTPUT_CSV_FILE, "a") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(line)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
