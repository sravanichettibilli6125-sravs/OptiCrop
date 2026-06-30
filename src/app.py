import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Opticrop - Smart Agriculture", layout="centered")

st.title("🌱 Opticrop: Intelligent Crop Recommendation")
st.write("Adjust the environmental factors below to discover the ideal crop recommendation.")

# 1. Load the trained classification models and label encoder safely
@st.cache_resource
def load_resources():
    try:
        rf = joblib.load("models/model_rf.pkl")
        lr = joblib.load("models/model_lr.pkl")
        kmeans = joblib.load("models/model_kmeans.pkl")
        encoder = joblib.load("models/label_encoder.pkl")
        return rf, lr, kmeans, encoder
    except FileNotFoundError:
        st.error("❌ Model files missing! Please run your training scripts first.")
        return None, None, None, None

rf_model, lr_model, kmeans_model, label_encoder = load_resources()

if rf_model and lr_model and label_encoder:
    # 2. Setup Sidebar Model Selector
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Classification Model", 
        ["Random Forest (Recommended)", "Logistic Regression"]
    )
    
    # 3. Create Input Fields for Soil & Weather features
    st.subheader("📊 Enter Environmental Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Nitrogen (N) content in soil", 0, 150, 90)
        p = st.slider("Phosphorus (P) content in soil", 5, 145, 42)
        k = st.slider("Potassium (K) content in soil", 5, 205, 43)
        ph = st.slider("Soil pH Value", 3.5, 10.0, 6.5, step=0.1)
    with col2:
        temp = st.slider("Temperature (°C)", 10.0, 50.0, 22.0, step=0.1)
        humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, 80.0, step=0.1)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 200.0, step=0.1)

    # Format inputs for model evaluation
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    input_df = pd.DataFrame(input_data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # 4. Make Predictions when the user hits the button
    if st.button("🔮 Recommend Best Crop", type="primary"):
        # Select the user's model preference
        selected_model = rf_model if "Random Forest" in model_choice else lr_model
        
        # Predict encoded number and decode to string text
        pred_encoded = selected_model.predict(input_df)[0]
        recommended_crop = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Predict environmental cluster using K-Means
        cluster_id = kmeans_model.predict(input_df)[0]
        
        # Display Results Beautifully
        st.success(f"### 🎉 Recommended Crop: **{recommended_crop.upper()}**")
        st.info(f"🤖 **Model Used:** {model_choice} | 🎯 **Environmental Cluster Assignment:** Cluster #{cluster_id}")