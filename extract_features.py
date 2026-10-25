import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model

# ============================================================
# VGG16 — Pré-entraîné sur ImageNet
# ============================================================
def extract_features_vgg16(X):
    base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model_vgg = Model(inputs=base_vgg.input, outputs=base_vgg.output)
    features_vgg = model_vgg.predict(X, batch_size=32, verbose=1)
    features_vgg = features_vgg.reshape(features_vgg.shape[0], -1)
    return features_vgg


# ============================================================
# ResNet50 — Pré-entraîné sur ImageNet
# ============================================================
def extract_features_resnet50(X):
    base_res = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model_res = Model(inputs=base_res.input, outputs=base_res.output)
    features_res = model_res.predict(X, batch_size=32, verbose=1)
    features_res = features_res.reshape(features_res.shape[0], -1)
    return features_res


# ============================================================
# AlexNet — Pré-entraîné sur ImageNet (via PyTorch)
# ============================================================
def extract_features_alexnet(X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    alexnet = models.alexnet(weights='DEFAULT')
    alexnet.eval()
    
    class FeatureExtractor(torch.nn.Module):
        def __init__(self, original_model):
            super(FeatureExtractor, self).__init__()
            self.features = original_model.features
            self.avgpool = original_model.avgpool
            self.classifier = torch.nn.Sequential(*list(original_model.classifier.children())[:-1])
        
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    model = FeatureExtractor(alexnet).to(device)
    model.eval()
    
    X_227 = np.array([cv2.resize(img, (227, 227)) for img in X])
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_size = 32
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(X_227), batch_size):
            batch = X_227[i:i+batch_size]
            
            batch_tensors = []
            for img in batch:
                img_uint8 = ((img * std) + mean)
                img_uint8 = np.clip(img_uint8, 0, 1)
                tensor = transform(img_uint8)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(device)
            features = model(batch_tensor)
            all_features.append(features.cpu().numpy())
    
    features_alex = np.concatenate(all_features, axis=0)
    return features_alex