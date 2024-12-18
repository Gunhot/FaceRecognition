#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Take picture from webcam

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
from MTCNN import create_mtcnn_net
from util.align_trans import *
import cv2
import argparse
from datetime import datetime
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'

cap = cv2.VideoCapture(0)

while cap.isOpened():

    isSuccess,frame = cap.read()
    if isSuccess:

        frame_text = cv2.putText(frame,'Press t to take a picture,q to quit.....',(10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("My Capture",frame_text)

    pressed = cv2.waitKey(1) & 0xFF
    if pressed == ord('t'):

        p =  frame

        data_path = Path(cur_path+'facebank')
        save_path = data_path / args.name
        if not save_path.exists():
            save_path.mkdir()

        try:
            bboxes, landmarks = create_mtcnn_net(p, 20, device,
                                                 p_model_path=cur_path+'Weights/pnet_Weights',
                                                 r_model_path=cur_path+'Weights/rnet_Weights',
                                                 o_model_path=cur_path+'Weights/onet_Weights')

            warped_face = Face_alignment(p, default_square=True, landmarks=landmarks)
            cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face[0])
        except:
            print('no face captured')

    elif pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()