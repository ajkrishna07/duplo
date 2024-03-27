import os, json, cv2, numpy as np
import csv
import sys

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate

CSV = sys.argv[1]
IMG = sys.argv[2]
COLOR = sys.argv[3]


def get_model(num_keypoints, weights_path=None):

    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0),
    )
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=True,
        num_keypoints=num_keypoints,
        num_classes=2,  # Background is the first class, object is the second class
        rpn_anchor_generator=anchor_generator,
    )

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = get_model(num_keypoints=4, weights_path="./" + COLOR + "_weights.pth")
model.to(device)

with open("./" + CSV + ".csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        img_name = "./" + IMG + "/" + row[0] + ".jpg"
        xmin = int(row[1]) - 5
        ymin = int(row[2]) - 5
        xmax = xmin + int(row[3]) + 5
        ymax = ymin + int(row[4]) + 5

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_patch = img[ymin:ymax, xmin:xmax]
        img_patch = F.to_tensor(img_patch)
        images = list([img_patch.to(device)])
        with torch.no_grad():
            model.to(device)
            model.eval()
            output = model(images)

        image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        scores = output[0]["scores"].detach().cpu().numpy()

        high_scores_idxs = np.where(scores > 0.7)[
            0
        ].tolist()  # Indexes of boxes with scores > 0.7
        post_nms_idxs = (
            torchvision.ops.nms(
                output[0]["boxes"][high_scores_idxs],
                output[0]["scores"][high_scores_idxs],
                0.3,
            )
            .cpu()
            .numpy()
        )  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

        keypoints = []
        for kps in (
            output[0]["keypoints"][high_scores_idxs][post_nms_idxs]
            .detach()
            .cpu()
            .numpy()
        ):
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        bboxes = []
        for bbox in (
            output[0]["boxes"][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        ):
            bboxes.append(list(map(int, bbox.tolist())))

        og_keypoints = [[kpt[0] + xmin, kpt[1] + ymin] for kpt in keypoints[0]]
        output_row = [row[0]]
        for pt in og_keypoints:
            output_row.append(str(pt[0]))
        for pt in og_keypoints:
            output_row.append(str(pt[1]))
        output_row.append("z")
        with open("./" + CSV + "_keypoints.csv", "a") as f2:
            writer = csv.writer(f2)
            writer.writerow(output_row)
