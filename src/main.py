import cv2
import torch
import os
import sys

from model import EmotionCNN
from config import EMOTION_CLASSES, IMG_SIZE, MODEL_SAVE_DIR, MODEL_NAME, DEVICE
from utils import preprocess_image_for_inference

def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train the model first using train.py or run_training.ipynb")
        return
    
    print("Loading model...")
    model = EmotionCNN(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded")
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Starting webcam...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            
            tensor = preprocess_image_for_inference(face_resized).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                _, predicted = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probs[0][predicted].item()
            
            emotion = EMOTION_CLASSES[predicted.item()]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Emotion Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done")

if __name__ == "__main__":
    main()
