#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import json
import numpy as np
import torch
from torchvision import transforms as trans
import cv2
from cv_bridge import CvBridge


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'
bridge = CvBridge()
transform = trans.Compose([trans.ToPILImage(), trans.ToTensor()])
configs = json.load(open(cur_path+"../Weights/fer2013_config.json"))
image_size=(configs["image_size"], configs["image_size"])


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

def emotion_recognition(b, image, model) :
    frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

    face = ensure_color(face)

    ## Error handling
    if len(face) == 0 : return None, None
    if face.shape[1] == 0 : return None, None
    face = cv2.resize(face, image_size)
    face = transform(face).cpu()
        
    face = torch.unsqueeze(face, dim=0) # size(1,3,224,224)
    # print(model(face))
    output = torch.squeeze(model(face), 0)
        
    proba = torch.softmax(output, 0)
    emo_proba, emo_idx = torch.max(proba, dim=0)       
    emo_idx = emo_idx.item()
    emo_proba = emo_proba.item()
    emo_label = FER_2013_EMO_DICT[emo_idx]
    cv2.putText(
            frame_bgr,
            "{} {}".format(emo_label, int(emo_proba * 100)),
            (int(b[2]), int(b[1]) + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
    return emo_label, emo_proba