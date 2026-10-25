import streamlit as st
import numpy as np
import cv2
import joblib
from extract_features import extract_features_vgg16, extract_features_resnet50, extract_features_alexnet
from PIL import Image

# ── Configuration page ───────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Style CSS personnalisé ───────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f8ff; }
    .stButton>button { background-color: #87CEEB; color: white; }
    .result-box { 
        background-color: #e6f3ff; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #4682B4;
    }
    .detail-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #b0d4f1;
        margin: 10px 0;
    }
    h1 { color: #2c5f8a; }
    h3 { color: #4682B4; }
    .caption { color: #6a9bc2; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des modèles ────────────────────────────
@st.cache_resource
def load_models():
    svm    = joblib.load('models/svm_model.pkl')
    dt     = joblib.load('models/dt_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return svm, dt, scaler

@st.cache_resource
def load_selector():
    return joblib.load('models/selector.pkl')

def preprocess_single(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224)).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean) / std

def predict(img_array):
    svm, dt, scaler = load_models()
    selector = load_selector()

    img = preprocess_single(img_array)
    X   = np.expand_dims(img, axis=0)

    f_alex   = extract_features_alexnet(X)
    f_vgg    = extract_features_vgg16(X)
    f_resnet = extract_features_resnet50(X)

    features = np.concatenate([f_alex, f_vgg, f_resnet], axis=1)
    features_sel = selector.transform(features)
    features_scaled = scaler.transform(features_sel)

    pred_svm = svm.predict(features_scaled)[0]
    pred_dt  = dt.predict(features_sel)[0]

    labels = {0: 'Happy', 1: 'Angry'}
    return labels[pred_svm], labels[pred_dt], features.shape[1], features_sel.shape[1]

# ── Interface ─────────────────────────────────────────
st.title("Emotion Recognition System")
st.markdown("<p class='caption'>Deep Learning-based facial expression classification</p>", unsafe_allow_html=True)



uploaded = st.file_uploader("Select an image file", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
    img_array = np.array(image)
    
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.image(image, width=300, caption="Input Image")
    
    with col_info:
        st.markdown("<div class='detail-box'>", unsafe_allow_html=True)
        st.markdown(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
        st.markdown(f"**Color mode:** RGB (3 channels)")
        st.markdown(f"**File format:** {uploaded.type}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.spinner("Running feature extraction and classification..."):
        pred_svm, pred_dt, total_features, selected_features = predict(img_array)
    
    st.divider()
    
    st.markdown("<h3>Classification Results</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("**SVM Classifier**")
        st.markdown(f"<h2 style='color: #2c5f8a; text-align: center;'>{pred_svm}</h2>", unsafe_allow_html=True)
    
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("**Decision Tree Classifier**")
        st.markdown(f"<h2 style='color: #2c5f8a; text-align: center;'>{pred_dt}</h2>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    
    
    st.markdown("<p class='caption' style='text-align: center;'>Models trained on 1,754 images (796 Happy, 958 Angry)</p>", unsafe_allow_html=True)