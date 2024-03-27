import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def compute_pose(x, y, plane):
    t_objp = np.array([[0, 0, 0], [0, 56, 0], [66, 0, 0], [66, 56, 0]], np.float32)
    t_imgp = np.array([[907, 1080], [395, 542], [1323, 605], [910, 185]], np.float32)

    mtx0 = cv2.initCameraMatrix2D([t_objp], [t_imgp], [1920, 1080])
    _, mtx, dist, r, t = cv2.calibrateCamera(
        [t_objp],
        [t_imgp],
        [1920, 1080],
        mtx0,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    xplane = np.array(
        [
            [-1.5, 3.0, 1.0],
            [-1.5, 0.0, 1.0],
            [-1.5, -3.0, 1.0],
            [-1.5, -3.0, -1.0],
            [-1.5, 0.0, -1.0],
            [-1.5, 3.0, -1.0],
        ],
        np.float32,
    )

    Xplane = np.array(
        [
            [1.5, 3.0, 1.0],
            [1.5, 0.0, 1.0],
            [1.5, -3.0, 1.0],
            [1.5, -3.0, -1.0],
            [1.5, 0.0, -1.0],
            [1.5, 3.0, -1.0],
        ],
        np.float32,
    )

    yplane = np.array(
        [
            [-1.5, -3.0, 1.0],
            [0.0, -3.0, 1.0],
            [1.5, -3.0, 1.0],
            [1.5, -3.0, -1.0],
            [0.0, -3.0, -1.0],
            [-1.5, -3.0, -1.0],
        ],
        np.float32,
    )

    Yplane = np.array(
        [
            [-1.5, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [1.5, 3.0, 1.0],
            [1.5, 3.0, -1.0],
            [0.0, 3.0, -1.0],
            [-1.5, 3.0, -1.0],
        ],
        np.float32,
    )

    zplane = np.array(
        [
            [1.5, 3.0, -1.0],
            [1.5, 0.0, -1.0],
            [1.5, -3.0, -1.0],
            [-1.5, -3.0, -1.0],
            [-1.5, 0.0, -1.0],
            [-1.5, 3.0, -1.0],
        ],
        np.float32,
    )

    Zplane = np.array(
        [
            [1.5, 3.0, 1.0],
            [1.5, 0.0, 1.0],
            [1.5, -3.0, 1.0],
            [-1.5, -3.0, 1.0],
            [-1.5, 0.0, 1.0],
            [-1.5, 3.0, 1.0],
        ],
        np.float32,
    )

    p1 = np.array([x[0], y[0]])
    p3 = np.array([x[1], y[1]])
    p2 = (p1 + p3) / 2.0
    p2 = p2.tolist()

    p4 = np.array([x[2], y[2]])
    p6 = np.array([x[3], y[3]])
    p5 = (p4 + p6) / 2.0
    p5 = p5.tolist()

    x.insert(1, p2[0])
    y.insert(1, p2[1])

    x.insert(4, p5[0])
    y.insert(4, p5[1])

    imgpoints = np.array([list(i) for i in zip(x, y)], np.float32)

    if plane == "x":
        _, rvecs, tvecs = cv2.solvePnP(xplane, imgpoints, mtx, dist, cv2.SOLVEPNP_SQPNP)
    elif plane == "y":
        _, rvecs, tvecs = cv2.solvePnP(yplane, imgpoints, mtx, dist, cv2.SOLVEPNP_SQPNP)
    elif plane == "z":
        _, rvecs, tvecs = cv2.solvePnP(zplane, imgpoints, mtx, dist, cv2.SOLVEPNP_SQPNP)
        return rvecs, tvecs, mtx, dist

    elif plane == "-x":
        _, rvecs, tvecs = cv2.solvePnP(Xplane, imgpoints, mtx, dist, cv2.SOLVEPNP_SQPNP)
    elif plane == "-y":
        _, rvecs, tvecs = cv2.solvePnP(Yplane, imgpoints, mtx, dist, cv2.SOLVEPNP_SQPNP)
    elif plane == "-z":
        _, rvecs, tvecs = cv2.solvePnP(Zplane, imgpoints, mtx, dist, cv2.SOLVEPNP_SQPNP)
        return rvecs, tvecs, mtx, dist


def draw_pose(rvecs, tvecs, mtx, dist, frame):
    box1 = np.float32(
        [[-1.5, 3.0, 1.0], [-1.5, -3.0, 1.0], [-1.5, -3.0, -1.0], [-1.5, 3.0, -1.0]]
    )
    imgpts1, jac = cv2.projectPoints(box1, rvecs, tvecs, mtx, dist)
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[0][0], np.int32)),
        tuple(np.array(imgpts1[1][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[1][0], np.int32)),
        tuple(np.array(imgpts1[2][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[2][0], np.int32)),
        tuple(np.array(imgpts1[3][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[3][0], np.int32)),
        tuple(np.array(imgpts1[0][0], np.int32)),
        (0, 255, 255),
        4,
    )

    box2 = np.float32(
        [[1.5, 3.0, 1.0], [1.5, -3.0, 1.0], [1.5, -3.0, -1.0], [1.5, 3.0, -1.0]]
    )
    imgpts2, jac = cv2.projectPoints(box2, rvecs, tvecs, mtx, dist)

    frame = cv2.line(
        frame,
        tuple(np.array(imgpts2[0][0], np.int32)),
        tuple(np.array(imgpts2[1][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts2[1][0], np.int32)),
        tuple(np.array(imgpts2[2][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts2[2][0], np.int32)),
        tuple(np.array(imgpts2[3][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts2[3][0], np.int32)),
        tuple(np.array(imgpts2[0][0], np.int32)),
        (0, 255, 255),
        4,
    )

    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[0][0], np.int32)),
        tuple(np.array(imgpts2[0][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[1][0], np.int32)),
        tuple(np.array(imgpts2[1][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[2][0], np.int32)),
        tuple(np.array(imgpts2[2][0], np.int32)),
        (0, 255, 255),
        4,
    )
    frame = cv2.line(
        frame,
        tuple(np.array(imgpts1[3][0], np.int32)),
        tuple(np.array(imgpts2[3][0], np.int32)),
        (0, 255, 255),
        4,
    )

    return frame


def anno_window(name, image_anno):
    x_vals = []
    y_vals = []

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_vals.append(x)
            y_vals.append(y)
            cv2.circle(image_anno, (x, y), 5, (255, 255, 0), -1)
            cv2.imshow(name, image_anno)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, 200, 200)
    cv2.setMouseCallback(name, draw_circle)
    cv2.imshow(name, image_anno)

    k_anno = cv2.waitKey(0)
    if k_anno == 13:
        cv2.destroyAllWindows()
        return x_vals, y_vals


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(annotated_image, detection_result):
    hand_landmarks_list = detection_result
    handedness_list = ["left", "right"]

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=float(hand_landmarks[i]),
                    y=float(hand_landmarks[i + 1]),
                    z=float(hand_landmarks[i + 2]),
                )
                for i in range(0, len(hand_landmarks), 3)
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [
            float(hand_landmarks[i]) for i in range(0, len(hand_landmarks), 3)
        ]
        y_coordinates = [
            float(hand_landmarks[i]) for i in range(1, len(hand_landmarks), 3)
        ]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image
