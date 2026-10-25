import cv2
import numpy as np
import joblib
import threading
import time
from extract_features import extract_features_vgg16, extract_features_resnet50, extract_features_alexnet

# ── Chargement des modèles ────────────────────────────
svm      = joblib.load('models/svm_model.pkl')
scaler   = joblib.load('models/scaler.pkl')
selector = joblib.load('models/selector.pkl')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Variables partagées entre les 2 fils ──────────────
label        = "Chargement..."
color        = (255, 255, 0)
latest_face  = None
lock         = threading.Lock()

def preprocess_face(face):
    if len(face.shape) == 2 or face.shape[2] == 1:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face = cv2.resize(face, (224, 224)).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (face - mean) / std

# ── Fil 2 : prédiction en arrière-plan ───────────────
def predict_loop():
    global label, color
    while True:
        with lock:
            face = latest_face.copy() if latest_face is not None else None
        
        if face is None:
            time.sleep(0.1)
            continue

        X = np.expand_dims(preprocess_face(face), axis=0)
        f_alex   = extract_features_alexnet(X)
        f_vgg    = extract_features_vgg16(X)
        f_resnet = extract_features_resnet50(X)

        features        = np.concatenate([f_alex, f_vgg, f_resnet], axis=1)
        features_sel    = selector.transform(features)
        features_scaled = scaler.transform(features_sel)

        pred  = svm.predict(features_scaled)[0]
        label = 'Happy 😊' if pred == 0 else 'Angry 😠'
        color = (0, 200, 0) if pred == 0 else (0, 0, 220)
        
        time.sleep(0.15)

# ── Lancer le fil de prédiction ───────────────────────
t = threading.Thread(target=predict_loop, daemon=True)
t.start()

# ── Fil 1 : webcam fluide ─────────────────────────────
cap = cv2.VideoCapture(0)
print("Webcam démarrée — appuie sur Q pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        with lock:
            latest_face = frame[y:y+h, x:x+w].copy()
        color_box = (0, 200, 0) if label == 'Happy 😊' else (0, 0, 220)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_box, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Reconnaissance Emotions — PFA', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()