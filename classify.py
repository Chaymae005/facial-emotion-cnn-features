import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

def classify(X, y, selector):
    # 1. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Normalisation pour SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3. SVM
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', 
          class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)

    # 4. Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # 5. Résultats
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_dt  = accuracy_score(y_test, y_pred_dt)

    print("=== RÉSULTATS ===")
    print(f"SVM            : {acc_svm:.4f} ({acc_svm*100:.2f}%)")
    print(f"Decision Tree  : {acc_dt:.4f}  ({acc_dt*100:.2f}%)")

    print("\n--- Rapport SVM ---")
    print(classification_report(y_test, y_pred_svm, target_names=['happy', 'angry']))

    print("\n--- Rapport Decision Tree ---")
    print(classification_report(y_test, y_pred_dt, target_names=['happy', 'angry']))

    # 6. Matrices de confusion
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, y_pred) in zip(axes,
        [('SVM', y_pred_svm), ('Decision Tree', y_pred_dt)]):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Happy', 'Angry'])
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'Matrice de confusion — {name}')

    plt.tight_layout()
    plt.savefig('resultats.png', dpi=150)
    plt.show()

    # 7. Sauvegarde des modèles
    joblib.dump(svm,      'models/svm_model.pkl')
    joblib.dump(dt,       'models/dt_model.pkl')
    joblib.dump(scaler,   'models/scaler.pkl')
    joblib.dump(selector, 'models/selector.pkl')

    print("\n✅ Modèles sauvegardés dans le dossier models/")