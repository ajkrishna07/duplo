import os
import sys
import cv2
import csv
import numpy as np
from corrector_utils import compute_pose, draw_pose, anno_window

ROOT = sys.argv[1]
IMG_FOLDER = sys.argv[2]
with open("./" + ROOT + ".csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        img_name = "./" + IMG_FOLDER + "/" + row[0] + ".jpg"
        x = [float(x) for x in row[1:5]]
        y = [float(y) for y in row[5:9]]
        plane = row[9]

        image = cv2.imread(img_name)

        cv2.namedWindow(row[0] + ".jpg", cv2.WINDOW_NORMAL)

        while True:
            rvecs, tvecs, mtx, dist = compute_pose(x.copy(), y.copy(), plane)
            image_pose = draw_pose(rvecs, tvecs, mtx, dist, image.copy())
            cv2.imshow(row[0] + ".jpg", image_pose)
            k = cv2.waitKey(0)
            if k == ord("f"):
                if plane == "z":
                    plane = "-z"
                elif plane == "-z":
                    plane = "z"
                cv2.destroyAllWindows()
            if k == ord("a"):
                x_vals, y_vals = anno_window(
                    "Annotating " + row[0] + ".jpg", image.copy()
                )
                if not x_vals or not y_vals:
                    continue
                x = [float(x) for x in x_vals]
                y = [float(y) for y in y_vals]
            if k == 3:
                with open("./" + ROOT + "_corrected.csv", "a") as f2:
                    writer = csv.writer(f2)
                    line = [row[0]]
                    for val in x:
                        line.append(str(int(val)))
                    for val in y:
                        line.append(str(int(val)))
                    line.append(plane)
                    writer.writerow(line)
                with open("./" + ROOT + "_pose_corrected.csv", "a") as f3:
                    writer = csv.writer(f3)
                    line = [row[0]]
                    for val in tvecs:
                        line.append(val[0])
                    rot_vecs, _ = cv2.Rodrigues(rvecs)
                    for angle in rot_vecs:
                        for val in angle:
                            line.append(val)
                    writer.writerow(line)
                break
            if k == ord("q"):
                exit()
    cv2.destroyAllWindows()
