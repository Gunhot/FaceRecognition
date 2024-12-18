import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from cv_bridge import CvBridge
from PIL import Image

from vgg import create_RepVGG_A0 as create

bridge = CvBridge()
cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'

# Load model
model = create(deploy=True)

# 8 Emotions
emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")

def init(device):
    # Initialise model
    global dev
    dev = device
    model.to(device)
    checkpoint = torch.load(cur_path+"../Weights/vgg.pth")
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}
    model.load_state_dict(ckpt)
    
    # Change to classify only 8 features
    model.linear.out_features = 8
    model.linear._parameters["weight"] = model.linear._parameters["weight"][:8,:]
    model.linear._parameters["bias"] = model.linear._parameters["bias"][:8]

    # Save to eval
    cudnn.benchmark = True
    model.eval()


def detect_emotion(images,conf=True):
    with torch.no_grad():
        # Normalise and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        x = torch.stack([transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])(Image.fromarray(image)) for image in images])
        # Feed through the model
        y = model(x.to('cpu'))
        result = []
        for i in range(y.size()[0]):
            # Add emotion to result
            emotion = (max(y[i]) == y[i]).nonzero().item()
            # Add appropriate label if required
            result.append([f"{emotions[emotion]}{f' ({100*y[i][emotion].item():.1f}%)' if conf else ''}",emotion])

        ## calculate latest 10 frame avg emotion
        # if len(recent_result) < 10 : recent_result.append(torch.Tensor.tolist(y[0]))
        # else : 
        #     recent_result.remove(recent_result[0])
        #     recent_result.append(torch.Tensor.tolist(y[0]))
        # temp_result = [0 for _ in range(len(emotions))]
        # for rr in recent_result :
        #     for i, e in enumerate(rr) :
        #         temp_result[i] += e
        # print(emotions[np.argmax(temp_result)])

    return result

def emotion_recognition_vgg(b, image) :
    try :
        frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
        face = frame_bgr[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
        img = torch.from_numpy(face).to('cpu')
        img = img.float()
        img = np.array(img.numpy())
        img = np.asarray(Image.fromarray(img.astype(np.uint8)))
        result = detect_emotion([img], True)
        result = result[0][0].replace('(', ' ').replace(')', ' ').replace('%', ' ').split(' ')
        result_emo = result[0]
        result_prob = float(result[2])/100
    except : 
        result_emo, result_prob = 'None', 0
    
    return result_emo, result_prob