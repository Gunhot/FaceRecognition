#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import dlib
import torch
import cv2
import numpy as np
from imutils import face_utils
from models.eyenet import EyeNet
from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample
from typing import List, Optional

# Set paths and device
cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'
dirname = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and weights
landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'Weights/shape_predictor_5_face_landmarks.dat'))
checkpoint = torch.load(cur_path + 'Weights/checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'], strict=False)

visualize = False

def detect_landmarks(b, frame):
    rectangle = dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)

def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]

        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale

        estimated_radius = 0.5 * eye_width * scale
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]

        transform_mat = center_mat * scale_mat * translate_mat
        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)
        if is_left:
            eye_image = np.fliplr(eye_image)
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=np.linalg.inv(transform_mat),
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes

def gaze(horizon, vertical):
    if horizon < 0.33:
        horizontal_position = 'left'
    elif horizon > 0.66:
        horizontal_position = 'right'
    else:
        horizontal_position = 'center'

    if vertical < 0.3:
        vertical_position = 'up'
    elif vertical > 0.6:
        vertical_position = 'down'
    else:
        vertical_position = 'center'

    if horizontal_position == 'center' and vertical_position == 'center':
        return 'center'
    elif horizontal_position != 'center' and vertical_position == 'center':
        return horizontal_position
    elif horizontal_position == 'center' and vertical_position != 'center':
        return vertical_position
    else:
        return f'{horizontal_position}-{vertical_position}'

def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            eye_img_array = np.array([eye.img], dtype=np.float32)
            x = torch.tensor(eye_img_array).to(device)
            _, landmarks = eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            landmarks = landmarks * np.array([oh/48, ow/80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks))
    return result

def smooth_eye_landmarks(eye: EyePrediction, prev_eye: Optional[EyePrediction], smoothing=0.2):
    if prev_eye is None:
        return eye
    return EyePrediction(
        eye_sample=eye.eye_sample,
        landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks)

def gaze_estimation(b, frame, gray_frame, visualize=False):
    landmarks_gaze = detect_landmarks(b, gray_frame)
    eye_samples = segment_eyes(gray_frame, landmarks_gaze)
    eye_preds = run_eyenet(eye_samples)
    
    left_eye = None
    right_eye = None
    lx = ly = rx = ry = None

    left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
    right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

    if left_eyes:
        left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
    if right_eyes:
        right_eye = smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)

    ratio = []
    ratio_2 = []
    for ep in [left_eye, right_eye]:
        if ep is None or ep.landmarks.size == 0:
            continue

        landmarks_array_x = []
        landmarks_array_y = []

        for i in range(17):
            for (x, y) in ep.landmarks[i:i+1]:
                landmarks_array_x.append(x)
                landmarks_array_y.append(y)
                if i == 32:
                    if ep.eye_sample.is_left:
                        lx = int(round(x))
                        ly = int(round(y))
                    else:
                        rx = int(round(x))
                        ry = int(round(y))

        if len(landmarks_array_x) == 0 or len(landmarks_array_y) == 0:
            continue

        width = abs(min(landmarks_array_x) - max(landmarks_array_x))
        height = abs(min(landmarks_array_y) - max(landmarks_array_y))
        if width == 0 or height == 0:
            continue
        
        center = (landmarks_array_x[-1], landmarks_array_y[-1])
        center_x = abs(center[0] - min(landmarks_array_x))
        center_y = abs(center[1] - min(landmarks_array_y))
        horizontal = center_x / width
        vertical = center_y / height
        ratio.append(horizontal)
        ratio_2.append(vertical)
        
    if len(ratio) > 0 and len(ratio_2) > 0:
        Eye = gaze(horizon=np.mean(ratio), vertical=np.mean(ratio_2))
    else:
        Eye = "unknown"

    # Draw rectangles around the eyes
    if left_eye and left_eye.landmarks.size > 0:
        left_eye_box = cv2.boundingRect(np.array([left_eye.landmarks[:, :2]], dtype=np.int32))
        cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), 
                      (left_eye_box[0] + left_eye_box[2], left_eye_box[1] + left_eye_box[3]), 
                      (255, 255, 0), 2)  # Cyan box for left eye
        for (x, y) in left_eye.landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)  # Yellow dots for left eye landmarks

    if right_eye and right_eye.landmarks.size > 0:
        right_eye_box = cv2.boundingRect(np.array([right_eye.landmarks[:, :2]], dtype=np.int32))
        cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), 
                      (right_eye_box[0] + right_eye_box[2], right_eye_box[1] + right_eye_box[3]), 
                      (255, 255, 0), 2)  # Cyan box for right eye
        for (x, y) in right_eye.landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)  # Yellow dots for right eye landmarks

    return Eye, lx, ly, rx, ry

def main():
    cap = cv2.VideoCapture(0)  # Start video capture from webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib's built-in HOG + SVM based face detector
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(gray_frame)

        for i, b in enumerate(faces):
            b = [b.left(), b.top(), b.right(), b.bottom()]
            Eye, lx, ly, rx, ry = gaze_estimation(b, frame, gray_frame, visualize=True)

            # Draw face bounding box
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)  # Yellow box for face

            # Draw gaze estimation results
            if Eye == "center":
                color = (0, 255, 0)  # Green for center
            elif Eye == "out of center":
                color = (0, 0, 255)  # Red for out of center
            else:
                color = (255, 0, 0)  # Blue for unknown
            cv2.putText(frame, Eye, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if lx is not None and ly is not None:
                cv2.circle(frame, (lx, ly), 3, (255, 0, 0), -1)
            if rx is not None and ry is not None:
                cv2.circle(frame, (rx, ry), 3, (255, 0, 0), -1)

        cv2.imshow("Gaze Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
