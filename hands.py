import os
import sys
import cv2
import csv
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from PIL import Image
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def process_hands(ROOT, IMG_FOLDER, fps):
    csv_file = "./" + ROOT + ".csv"
    out_data = pd.DataFrame([])
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        lines = []
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the video mode:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.9,
            min_tracking_confidence=0.9,
            num_hands=2,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        for row in reader:
            frame_num = row[0]
            line = [int(frame_num)]
            image = "./" + IMG_FOLDER + "/" + str(frame_num) + ".jpg"
            frame = mp.Image.create_from_file(image)
            timestamp = int(float(frame_num) / fps * 1000)
            hand_landmarker_result = detector.detect_for_video(frame, timestamp)
            hands = hand_landmarker_result.hand_landmarks
            handedness_list = hand_landmarker_result.handedness
            if len(hands) == 1:
                if handedness_list[0][0].category_name == "Left":
                    for landmark in hands[0]:
                        line.append(landmark.x)
                        line.append(landmark.y)
                        line.append(landmark.z)
                    for i in range(63):
                        line.append(np.nan)
                elif handedness_list[0][0].category_name == "Right":
                    for i in range(63):
                        line.append(np.nan)
                    for landmark in hands[0]:
                        line.append(landmark.x)
                        line.append(landmark.y)
                        line.append(landmark.z)
            elif len(hands) == 2:
                if (
                    handedness_list[0][0].category_name
                    != handedness_list[1][0].category_name
                ):
                    if handedness_list[0][0].category_name == "Left":
                        for landmark in hands[0]:
                            line.append(landmark.x)
                            line.append(landmark.y)
                            line.append(landmark.z)
                        for landmark in hands[1]:
                            line.append(landmark.x)
                            line.append(landmark.y)
                            line.append(landmark.z)
                    elif handedness_list[0][0].category_name == "Right":
                        for landmark in hands[1]:
                            line.append(landmark.x)
                            line.append(landmark.y)
                            line.append(landmark.z)
                        for landmark in hands[0]:
                            line.append(landmark.x)
                            line.append(landmark.y)
                            line.append(landmark.z)
                else:
                    for landmark in hands[0]:
                        line.append(landmark.x)
                        line.append(landmark.y)
                        line.append(landmark.z)
                    for landmark in hands[1]:
                        line.append(landmark.x)
                        line.append(landmark.y)
                        line.append(landmark.z)
            else:
                for i in range(126):
                    line.append(np.nan)
            lines.append(line)
        lines_pd = pd.DataFrame(lines)
        lines_pd_inter = lines_pd.interpolate(limit_direction="both")
        out_data = pd.concat([out_data, lines_pd_inter], axis=0)
    out_data.to_csv("./" + ROOT + "_hands.csv", index=False)


ROOT = sys.argv[1]
IMG_FOLDER = sys.argv[2]
VIDEO = sys.argv[3]

cap = cv2.VideoCapture("./" + VIDEO + ".mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

process_hands(ROOT, IMG_FOLDER, fps)
