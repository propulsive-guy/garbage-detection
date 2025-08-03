import streamlit as st
import requests
from PIL import Image
import io

# 🔐 Authentication token - keep it secret!
AUTH_TOKEN = "pugarch123"

# 🌐 GCP-hosted Flask API endpoint
API_URL = "https://public-pugarch.de.r.appspot.com/predict"

# 🖥️ Streamlit UI setup
st.set_page_config(page_title="YOLOv8 Detector", layout="centered")
st.title("🧠 YOLOv8 Object Detection")
st.write("Upload an image, and the model will detect objects in it.")

# 📤 Image uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Processing..."):
            try:
                # 🔐 Prepare headers with auth token
                headers = {
                    "Authorization": f"Bearer {AUTH_TOKEN}"
                }

                # 📡 Send image to API
                response = requests.post(
                    API_URL,
                    headers=headers,
                    files={"image": uploaded_file}
                )

                if response.status_code == 200:
                    # 🖼️ Read and show image from response
                    result_img = Image.open(io.BytesIO(response.content))
                    st.success("Detection Complete ✅")
                    st.image(result_img, caption="Detected Image", use_column_width=True)
                else:
                    st.error(f"❌ Server Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")
