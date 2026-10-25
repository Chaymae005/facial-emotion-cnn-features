import numpy as np
from extract_features import extract_features_vgg16, extract_features_resnet50, extract_features_alexnet
from concatenate_features import concatenate_features
from select_features import select_kbest
from classify import classify
import os

os.makedirs('models', exist_ok=True)

# ── Étape 1 : Chargement des données ─────────────────
print(">>> Étape 1 : Chargement des données...")
from load_data import load_dataset

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HA")
X, y = load_dataset(data_path)

# ── Étape 2 : Extraction des features ────────────────
print("\n>>> Étape 2 : Extraction des features avec VGG16...")
f_vgg = extract_features_vgg16(X)

print("\n>>> Étape 3 : Extraction des features avec ResNet50...")
f_resnet = extract_features_resnet50(X)

print("\n>>> Étape 4 : Extraction des features avec AlexNet...")
f_alex = extract_features_alexnet(X)

# ── Étape 5 : Concaténation ───────────────────────────
print("\n>>> Étape 5 : Concaténation des features...")
features_combined = concatenate_features(f_alex, f_vgg, f_resnet)

# Sauvegarde des features combinées
np.save('models/features_combined.npy', features_combined)
np.save('models/labels.npy', y)
print("Features et labels sauvegardés dans models/")

# ── Étape 6 : Sélection ──────────────────────────────
print("\n>>> Étape 6 : Sélection des features...")
X_selected, selector = select_kbest(features_combined, y, k=1000)

# ── Étape 7 : Classification ───────────────────────────
print("\n>>> Étape 7 : Classification et comparaison...")
classify(X_selected, y, selector)