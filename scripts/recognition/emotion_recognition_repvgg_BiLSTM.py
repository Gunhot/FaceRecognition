import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import tensorflow as tf
from tensorflow import keras

from cv_bridge import CvBridge
from PIL import Image

from vgg import create_RepVGG_A0 as create

bridge = CvBridge()
cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'

# Load RepVGG model
model_RepVGG = create(deploy=True)
model_BiLSTM = None

# 8 Emotions
# emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")
OuluCASIA_emotions = ("anger","disgust","fear","happy","sad","surprise")
RAVDESS_emotions = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised')
# dataset choose   # 0:OuluCASIA, 1:RAVDESS
dataset = 0

# cur_seq = torch.empty((0, 1280), dtype=torch.float32)
cur_seq = np.empty((0, 1280))
cur_seq_len = 0

def init(device):
    # Initialise model
    global dev
    dev = device

    ## RepVGG

    model_RepVGG.to(device)
    Weight_RepVGG = torch.load(cur_path+"../Weights/vgg.pth")
    if 'state_dict' in Weight_RepVGG:
        Weight_RepVGG = Weight_RepVGG['state_dict']
    ckpt = {k.replace('module.', ''):v for k,v in Weight_RepVGG.items()}
    model_RepVGG.load_state_dict(ckpt)
    
    # Change to classify only 8 features
    model_RepVGG.linear.out_features = 8
    model_RepVGG.linear._parameters["weight"] = model_RepVGG.linear._parameters["weight"][:8,:]
    model_RepVGG.linear._parameters["bias"] = model_RepVGG.linear._parameters["bias"][:8]
    model_RepVGG.linear = nn.Identity()

    # Save to eval
    cudnn.benchmark = True
    model_RepVGG.eval()

    ## Bi-LSTM
    global model_BiLSTM
    if dataset == 0:
        model_BiLSTM = tf.keras.models.load_model(cur_path+"../Weights/best_ckpt_OuluCasia.h5")
    else :
        model_BiLSTM = tf.keras.models.load_model(cur_path+"../Weights/best_ckpt_RAVDESS3.h5")
        # best_ckpt_RAVDESS  : 모든 frame으로 training
        # best_ckpt_RAVDESS3 : 3frame마다 1 image만 training에 사용
    print("Model Created")


def detect_emotion(images,conf=True):
    with torch.no_grad():
        # Normalise and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        x = torch.stack([transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])(Image.fromarray(image)) for image in images])
        # Feed through the model
        y = model_RepVGG(x.to('cpu')).detach().numpy()
        global cur_seq, cur_seq_len
        cur_seq = np.append(cur_seq, y, axis=0)

        # WINDOWING # window_size = 10, window_step = 2
        if cur_seq.shape[0] >= 11 : cur_seq = cur_seq[2:]
        if cur_seq.shape[0]%2 != 0 and cur_seq.shape[0] >= 2 : return None

        temp_seq = np.array([np.copy(cur_seq)])

        # print("cur seq shape : ", temp_seq.shape)
        predicted = torch.from_numpy(model_BiLSTM.predict(temp_seq, verbose=0))
        result = []
        for i in range(predicted.size()[0]):
            # Add emotion to result
            emotion = (max(predicted[i]) == predicted[i]).nonzero().item()
            # Add appropriate label if required
            if dataset == 0 :
                result.append([f"{OuluCASIA_emotions[emotion]}{f' ({100*predicted[i][emotion].item():.1f}%)' if conf else ''}",emotion])
            else : 
                result.append([f"{RAVDESS_emotions[emotion]}{f' ({100*predicted[i][emotion].item():.1f}%)' if conf else ''}",emotion])
    return result

prev_emo = None
prev_prob = None
def emotion_recognition_vgg(b, image) :
    # thres = 0.35
    global prev_emo, prev_prob

    try :
        frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
        face = frame_bgr[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
        img = torch.from_numpy(face).to('cpu')
        img = img.float()
        img = np.array(img.numpy())
        img = np.asarray(Image.fromarray(img.astype(np.uint8)))
        result = detect_emotion([img], True)
        if result == None :
            return prev_emo, prev_prob
        result = result[0][0].replace('(', ' ').replace(')', ' ').replace('%', ' ').split(' ')
        result_emo = result[0]
        result_prob = float(result[2])/100

        # if result_prob < thres :
        #     result_emo = 'neutral'
    except : 
        result_emo, result_prob = 'None', 0
    
    prev_emo = result_emo
    prev_prob = result_prob
    return result_emo, result_prob