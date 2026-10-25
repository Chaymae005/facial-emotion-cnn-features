import os
import cv2
import numpy as np

def load_dataset(data_path, target_size=(224, 224)):
    X, y = [], []
    classes = {'happy': 0, 'angry': 1}
    
    # Paramètres de normalisation ImageNet (attendus par VGG16 et ResNet50)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    for label, idx in classes.items():
        folder = os.path.join(data_path, label)
        
        # Vérification que le dossier existe
        if not os.path.exists(folder):
            print(f"⚠️ Dossier non trouvé : {folder}")
            continue
            
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                # 1. BGR → RGB (cv2.imread charge en BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 2. Redimensionnement
                img = cv2.resize(img, target_size)
                
                # 3. Conversion float32 et normalisation [0, 1]
                img = img.astype('float32') / 255.0
                
                # 4. Standardisation ImageNet
                img = (img - mean) / std
                
                X.append(img)
                y.append(idx)
    
    X, y = np.array(X, dtype=np.float32), np.array(y)
    
    # Vérification de l'équilibre des classes
    n_happy = np.sum(y == 0)
    n_angry = np.sum(y == 1)
    print(f"\n📊 Répartition des classes :")
    print(f"   Happy : {n_happy} images ({n_happy/len(y)*100:.1f}%)")
    print(f"   Angry : {n_angry} images ({n_angry/len(y)*100:.1f}%)")
    
    return X, y


# ==== EXÉCUTION ====
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HA")

X, y = load_dataset(data_path)
print(f"\n✅ Images chargées : {X.shape}")
print(f"✅ Labels          : {y.shape}")
print(f"✅ Type données    : {X.dtype}")
print(f"✅ Valeurs min/max : {X.min():.3f} / {X.max():.3f}")