import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import dlib

# Load models and define configurations
from vgg import create_RepVGG_A0 as create
import os

# Set paths and device
cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'
dirname = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define emotions
OuluCASIA_emotions = ("anger", "disgust", "fear", "happy", "sad", "surprise")
RAVDESS_emotions = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised')
dataset = 0  # 0:OuluCASIA, 1:RAVDESS

# Initialize sequences
cur_seq = np.empty((0, 1280))
cur_seq_len = 0

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def init(device):
    global dev, model_RepVGG, model_BiLSTM
    dev = device

    # Initialize RepVGG model
    model_RepVGG = create(deploy=True)
    model_RepVGG.to(device)
    Weight_RepVGG = torch.load(cur_path + "Weights/vgg.pth")
    if 'state_dict' in Weight_RepVGG:
        Weight_RepVGG = Weight_RepVGG['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in Weight_RepVGG.items()}
    model_RepVGG.load_state_dict(ckpt)

    # Modify the last layer for 8 output classes
    model_RepVGG.linear.out_features = 8
    model_RepVGG.linear._parameters["weight"] = model_RepVGG.linear._parameters["weight"][:8, :]
    model_RepVGG.linear._parameters["bias"] = model_RepVGG.linear._parameters["bias"][:8]
    model_RepVGG.linear = nn.Identity()

    # Set model to evaluation mode
    cudnn.benchmark = True
    model_RepVGG.eval()

    # Load Bi-LSTM model
    if dataset == 0:
        model_BiLSTM = tf.keras.models.load_model(cur_path + "Weights/best_ckpt_OuluCasia.h5")
    else:
        model_BiLSTM = tf.keras.models.load_model(cur_path + "Weights/best_ckpt_RAVDESS3.h5")
    print("Models loaded and initialized.")

def detect_emotion(images, conf=True):
    with torch.no_grad():
        # Normalize and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        x = torch.stack([transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])(Image.fromarray(image)) for image in images])
        # Feed through RepVGG model
        y = model_RepVGG(x.to(dev)).detach().cpu().numpy()
        global cur_seq, cur_seq_len
        cur_seq = np.append(cur_seq, y, axis=0)

        # WINDOWING # window_size = 10, window_step = 2
        if cur_seq.shape[0] >= 11:
            cur_seq = cur_seq[2:]
        if cur_seq.shape[0] % 2 != 0 and cur_seq.shape[0] >= 2:
            return None

        temp_seq = np.array([np.copy(cur_seq)])
        predicted = torch.from_numpy(model_BiLSTM.predict(temp_seq, verbose=0))
        result = []
        for i in range(predicted.size()[0]):
            emotion = (max(predicted[i]) == predicted[i]).nonzero().item()
            if dataset == 0:
                result.append([f"{OuluCASIA_emotions[emotion]}{f' ({100 * predicted[i][emotion].item():.1f}%)' if conf else ''}", emotion])
            else:
                result.append([f"{RAVDESS_emotions[emotion]}{f' ({100 * predicted[i][emotion].item():.1f}%)' if conf else ''}", emotion])
    return result

def draw_landmarks(frame, landmarks):
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

prev_emo = None
prev_prob = None

def emotion_recognition_vgg(bbox, frame):
    global prev_emo, prev_prob
    try:
        face = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        result = detect_emotion([face], True)
        if result is None:
            return prev_emo, prev_prob
        result = result[0][0].replace('(', ' ').replace(')', ' ').replace('%', ' ').split(' ')
        result_emo = result[0]
        result_prob = float(result[2]) / 100
    except:
        result_emo, result_prob = 'None', 0

    prev_emo = result_emo
    prev_prob = result_prob
    return result_emo, result_prob

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            bbox = [x1, y1, x2, y2]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Predict emotion
            emotion, confidence = emotion_recognition_vgg(bbox, frame)
            if emotion:
                cv2.putText(frame, f'{emotion}: {confidence*100:.1f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Detect facial landmarks
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)

        # Display the frame
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
