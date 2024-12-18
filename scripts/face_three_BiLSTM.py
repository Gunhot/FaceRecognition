#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: AIRocker
"""

import sys
import time

import os
import json
import rospy
import torch
import cv2

from models import densenet121

sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import argparse
from torchvision import transforms as trans
from util.align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet

from recognition.gaze_estimation import *
from recognition.emotion_recognition_repvgg_BiLSTM import *
from recognition.face_recognition import *

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg
from vision_msg.msg import face_recognition_result
from std_msgs.msg import String


cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'
device = torch.device("cpu")
configs = json.load(open(cur_path+"Weights/fer2013_config.json"))
image_size=(configs["image_size"], configs["image_size"])
model_face_detection = densenet121(in_channels=3, num_classes=7)
model_face_detection.cpu()


# CV bridge : connection between OpenCV and ROS
bridge = CvBridge()

# initialize result publisher
result_pub = rospy.Publisher('face_recognition_result', face_recognition_result, queue_size=10)

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized


def message_callback(message):
    global flag
    if message == String("On"):
        flag = 1
    else:
        flag = 0        


def camera_callback():

    # Get ROS image using wait for message
    #image = rospy.wait_for_message('/camera/color/image_raw', Image_msg)
    image = rospy.wait_for_message('/usb_cam/image_raw', Image_msg)
    frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
    start_time = time.time()
    input = resize_image(frame_bgr, args.scale)
    bboxes, landmarks = [], []
    try:
        bboxes, landmarks = create_mtcnn_net(input, args.mini_face, device, 
                                            p_model_path=cur_path+'Weights/pnet_Weights',
                                            r_model_path=cur_path+'Weights/rnet_Weights',
                                            o_model_path=cur_path+'Weights/onet_Weights')
    except:
        pass

    embs = []
    Eye = None
    emo_result_list = []
    gaze_list = []

    if len(bboxes) != 0:
        bboxes = bboxes / args.scale
        landmarks = landmarks / args.scale
    
    faces = Face_alignment(frame_bgr, default_square=True, landmarks=landmarks) 


    for img in faces:  
            embs.append(detect_model(test_transform(img).to(device).unsqueeze(0)))
        
    if len(embs) >= 1:

        # Face Recognition
        if args.face_recognition :
            results, score = face_recognition(embs, targets, args.threshold, names)
            # convert distance to score dis(0.7,1.2) to score(100,60)
            score_100 = torch.clamp(score*-80+156,0,100)
            names[0] = 'unknown'


        for i, b in enumerate(bboxes):
            # Face Detection - Put Face Bounding box
            if args.visualize :
                frame_bgr = cv2.rectangle(frame_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,0,0), 1)

            # Face Recognition
            if args.face_recognition and args.visualize :
                # Put Text about name and score
                frame_bgr = cv2.putText(frame_bgr, names[results[i] + 1]+" / Score : {:.0f}".format(score_100[i]),
                    (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255)) 
            
            
            # Facial Expression Recognition
            if args.emotion_recognition :
                emo_label, emo_proba = emotion_recognition_vgg(b, image)
                if emo_label == None : continue
                if args.visualize :
                    frame_bgr = cv2.putText(frame_bgr, "{} {}".format(emo_label, int(emo_proba * 100)),
                        (int(b[2]), int(b[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                emo_result_list.append([emo_label, int(emo_proba * 100)])

            # Gaze Estimation
            if args.gaze_estimation :
                Eye, lx, ly, rx, ry = gaze_estimation(b, i, image, args.visualize)
                if args.visualize :
                    cv2.circle(frame_bgr, (rx, ry), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame_bgr, (lx, ly), 1, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                gaze_list.append(Eye)
    
    if args.visualize :
        # Calculate FPS
        FPS = 1.0 / (time.time() - start_time)
        frame_bgr = cv2.putText(frame_bgr, Eye, (90,60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        frame_bgr = cv2.putText(frame_bgr,'FPS: {:.1f}'.format(FPS),(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0))
            
        cv2.imshow('video', frame_bgr)

    if cv2.waitKey(1)&0xFF == ord('q'):
        cv2.destroyAllWindows()
        sys.exit("Terminated")

    # Topic Publish
    try:
        fr_result = face_recognition_result()
        
        fr_result.num_face = str(len(bboxes))
        
        for num, res in enumerate(bboxes):
            #print(res)
            fr_result.names += names[results[num] + 1] +' '
            fr_result.face_xmin += str(int(res[0])) + ' '
            fr_result.face_ymin += str(int(res[1])) + ' '
            fr_result.face_xmax += str(int(res[2])) + ' '
            fr_result.face_ymax += str(int(res[3])) + ' '
            fr_result.face_score += str(int(score_100[num].item())) + ' '
            fr_result.emotion += str(emo_result_list[num][0]) + ' '
            fr_result.emotion_score += str(emo_result_list[num][1]) + ' '
            fr_result.gaze_center += str(gaze_list[num]) + ' '

        result_pub.publish(fr_result)
    
    except:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face recognition demo')
    parser.add_argument('-cpu','--cpu',help='force cpu',default=False, action="store_true")
    parser.add_argument('-th','--threshold',help='threshold score to decide identical faces',default=60, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true", default= False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true", default= False)
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true",default= True )
    parser.add_argument("--scale", dest='scale', help="input frame scale to accurate the speed", default=0.5, type=float)
    parser.add_argument('--mini_face', dest='mini_face', help="Minimum face to be detected. derease to increase accuracy. Increase to increase speed", default=40, type=int)
    parser.add_argument('-f','--force', help='execution without rostopic publish', action="store_true", default= False)  
    parser.add_argument('-v','--visualize', help='visualize the result', action="store_true", default= False)  

    # option for result type - face detection is default
    parser.add_argument('-fr','--face_recognition', help='face recognition', action="store_true", default= False)
    parser.add_argument('-er','--emotion_recognition', help='facial expression recognition', action="store_true", default= False)
    parser.add_argument('-ge','--gaze_estimation', help='gaze estimation', action="store_true", default= False)

    args = parser.parse_args()

    # set device using only cpu
    device = torch.device("cpu")


    # set detect model
    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    if args.cpu :
        detect_model.load_state_dict(torch.load(cur_path+'Weights/MobileFace_Net', map_location='cpu'))
    else :
        detect_model.load_state_dict(torch.load(cur_path+'Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()


    # load model
    init('cpu')
    if args.cpu :
        state = torch.load(cur_path+'Weights/densenet121_test_2022Mar15_02.42', map_location='cpu')
    else :
        state = torch.load(cur_path+'Weights/densenet121_test_2022Mar15_02.42')
    model_face_detection.load_state_dict(state["net"])
    model_face_detection.eval()

    # load facebank data for face recognition
    if args.face_recognition :
        targets, names = load_facebank_data()
    
    # Set transform 
    test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    # ros node 
    rospy.init_node('face_recognition', anonymous = True)
    
    global flag
    flag = 0
    
    while True:
        message_sub = rospy.Subscriber('/face_recognition_msg', String, message_callback)
        if args.force : flag = 1
        if flag == 1:
            camera_callback()
    cv2.destroyAllWindows()
