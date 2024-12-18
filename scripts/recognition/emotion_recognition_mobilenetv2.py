import os
import numpy as np
from cv_bridge import CvBridge
from tensorflow import keras
from skimage.transform import resize

bridge = CvBridge()
cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'

model = keras.models.load_model(cur_path + '../Weights/MobileNetV2_FER2013.h5')

# 8 Emotions
Emotions = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def emotion_recognition_mobilenetv2(b, image) :
    frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
    face = frame_bgr[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    img = np.array(face)
    img = resize(img, (224, 224, 3))
    img = np.array([img])
    # img = img/255.0
    predict_class = model.predict(img, verbose = 0)
    predict_class = predict_class[0]
    result_emo = Emotions[np.argmax(predict_class)]
    result_prob = max(predict_class)

    
    return result_emo, result_prob