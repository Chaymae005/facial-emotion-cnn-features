import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def select_kbest(X, y, k=1000):
    """
    Sélectionne les k meilleures features selon le test F (ANOVA).
    Garde uniquement les features les plus corrélées avec la variable cible.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Avant sélection : {X.shape}")
    print(f"Après sélection : {X_selected.shape}")
    
    return X_selected, selector

def select_pca(X, n_components=500):
    """
    Réduction de dimension par PCA (Analyse en Composantes Principales).
    Alternative à SelectKBest, non utilisée dans le pipeline principal.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    variance = sum(pca.explained_variance_ratio_)
    print(f"Variance conservée : {variance:.2%}")
    
    return X_pca, pca