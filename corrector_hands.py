import os
import sys
import cv2
import csv
import numpy as np
from corrector_utils import draw_landmarks_on_image, anno_window

ROOT = sys.argv[1]
IMG_FOLDER = sys.argv[2]
with open("./" + ROOT + ".csv", "r") as f:
    reader = csv.reader(f)
    reader = iter(reader)
    next(reader)
    for row in reader:
        img_name = "./" + IMG_FOLDER + "/" + row[0] + ".jpg"
        left = row[1:64]
        right = row[64:]

        image = cv2.imread(img_name)
        image = image[180:801, 800:1581]
        height, width, _ = image.shape

        cv2.namedWindow(row[0] + ".jpg", cv2.WINDOW_NORMAL)

        while True:
            anno_image = draw_landmarks_on_image(image.copy(), [left, right])
            cv2.imshow(row[0] + ".jpg", anno_image)
            k = cv2.waitKey(0)
            if k == ord("f"):
                left, right = right, left
                cv2.destroyAllWindows()
            if k == ord("l"):
                x_vals, y_vals = anno_window(
                    "Annotating left hand in " + row[0] + ".jpg", image.copy()
                )
                if not x_vals or not y_vals:
                    continue
                x = [float(x) / float(width) for x in x_vals]
                y = [float(y) / float(height) for y in y_vals]
                idx = 0
                for i in range(0, len(left), 3):
                    left[i] = x[idx]
                    idx += 1
                idx = 0
                for i in range(1, len(left), 3):
                    left[i] = y[idx]
                    idx += 1
            if k == ord("r"):
                x_vals, y_vals = anno_window(
                    "Annotating right hand in " + row[0] + ".jpg", image.copy()
                )
                x = [float(x) / float(width) for x in x_vals]
                y = [float(y) / float(height) for y in y_vals]
                idx = 0
                for i in range(0, len(right), 3):
                    right[i] = x[idx]
                    idx += 1
                idx = 0
                for i in range(1, len(right), 3):
                    right[i] = y[idx]
                    idx += 1
            if k == 3:
                with open(
                    "./" + ROOT + "_" + TOWER + "_hands_corrected.csv", "a"
                ) as f2:
                    writer = csv.writer(f2)
                    line = [row[0]]
                    for val in left:
                        line.append(str(val))
                    for val in right:
                        line.append(str(val))
                    writer.writerow(line)
                break
            if k == ord("q"):
                exit()

    cv2.destroyAllWindows()
