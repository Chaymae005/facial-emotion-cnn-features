import numpy as np

def concatenate_features(f_alexnet, f_vgg16, f_resnet50):
    """
    Concatène les features extraites par trois CNN pré-entraînés :
    AlexNet, VGG16 et ResNet50.
    
    Principe : approche multi-vues (multi-view feature fusion).
    Chaque extracteur capture des patterns différents, leur combinaison
    crée une représentation plus riche et discriminative.
    """
    features_combined = np.concatenate(
        [f_alexnet, f_vgg16, f_resnet50],
        axis=1
    )
    
    # Affichage des dimensions pour vérification
    print(f"AlexNet  : {f_alexnet.shape}")
    print(f"VGG16    : {f_vgg16.shape}")
    print(f"ResNet50 : {f_resnet50.shape}")
    print(f"Total combiné : {features_combined.shape}")
    
    return features_combined