import cv2
import numpy as np

def preprocess_dataset(X, target_size=(224, 224)):
    X_preprocessed = []
    
    for img in X:
        # 1. Convertir en RGB si l'image est en niveaux de gris
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 2. Redimensionner à 224x224
        img = cv2.resize(img, target_size)
        
        # 3. Normaliser les pixels (0-255 → 0-1)
        img = img.astype('float32') / 255.0
        
        # 4. Normalisation ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        
        X_preprocessed.append(img)
    
    return np.array(X_preprocessed)