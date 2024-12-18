#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import dlib
import torch
import cv2
from cv_bridge import CvBridge
import numpy as np
from imutils import face_utils
from models.eyenet import EyeNet
from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample
from typing import List, Optional

cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'
dirname = os.path.dirname(__file__)
device = torch.device("cpu")
bridge = CvBridge()

landmarks_detector = dlib.shape_predictor(os.path.join(dirname, '../Weights/shape_predictor_5_face_landmarks.dat'))
checkpoint = torch.load(cur_path+'../Weights/checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'] , strict = False)

visualize = False


def detect_landmarks(b, frame, scale_x=0, scale_y=0):
    rectangle = dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    
    face_landmarks = landmarks_detector(frame, rectangle)
    
    return face_utils.shape_to_np(face_landmarks)


### gaze estimation fuction ###
def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)
        if is_left:
            eye_image = np.fliplr(eye_image)
            if visualize:
                cv2.imshow('left eye image', eye_image)
        else:
            if visualize:
                cv2.imshow('right eye image', eye_image)
        
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes



def gaze(horizon,vertical):
    if horizon<=0.33:
        x = 'out of center'
    elif horizon>= 0.66:
        x = 'out of center'
    elif vertical<= 0.3:
        x = 'out of center'
    elif vertical>=0.7:
        x = 'out of center'
    else:
        x = 'center'
    return x

def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(device)
            _, landmarks= eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            # gaze = np.asarray(gaze.cpu().numpy()[0])
            # assert gaze.shape == (2,)
            assert landmarks.shape == (34, 2)

            landmarks = landmarks * np.array([oh/48, ow/80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            assert landmarks.shape == (34, 3)
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            assert landmarks.shape == (34, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks)) # gaze=gaze))
    return result

def smooth_eye_landmarks(eye: EyePrediction, prev_eye: Optional[EyePrediction], smoothing=0.2): # , gaze_smoothing=0.4):
    if prev_eye is None:
        return eye
    return EyePrediction(
        eye_sample=eye.eye_sample,
        landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks)
        # gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze)


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

def gaze_estimation(b, i, image, v):
    global visualize
    visualize = v
    if v:
        frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    alpha = 0.95
    landmarks_gaze = None
    left_eye = None
    right_eye = None
    lx = None
    ly = None
    rx = None
    ry = None
    next_landmarks = detect_landmarks(b, gray)

    if landmarks_gaze is not None:
        landmarks_gaze = next_landmarks * alpha + (1 - alpha) * landmarks_gaze
    else:
        landmarks_gaze = next_landmarks

    if landmarks_gaze is not None:
        eye_samples = segment_eyes(gray, landmarks_gaze)

        eye_preds = run_eyenet(eye_samples)
        left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
        right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

        if left_eyes:
            left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
        if right_eyes:
            right_eye = smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)

        ratio=[]
        ratio_2=[]
        for ep in [left_eye, right_eye]:
            landmarks_array_x = []
            landmarks_array_y = []
                 
            for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32]:
                for (x, y) in ep.landmarks[i:i+1]:
                    landmarks_array_x.append(x)
                    landmarks_array_y.append(y)
                    if i == 32:
                        if ep.eye_sample.is_left : 
                            lx = int(round(x))
                            ly = int(round(y))
                        else : 
                            rx = int(round(x))
                            ry = int(round(y))

            width=abs(min(landmarks_array_x)-max(landmarks_array_x))
            height=abs(min(landmarks_array_y)-max(landmarks_array_y))
            center=(landmarks_array_x[-1],landmarks_array_y[-1])
            center_x=abs(center[0]-min(landmarks_array_x))
            center_y=abs(center[1]-min(landmarks_array_y))
            horizontal = center_x / width
            vertical = center_y / height
            ratio.append(horizontal)
            ratio_2.append(vertical)
        Eye = gaze(horizon=np.mean(ratio),vertical=np.mean(ratio_2))
    return Eye, lx, ly, rx, ry
