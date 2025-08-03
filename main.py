from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
from functools import wraps

app = Flask(__name__)

AUTH_TOKEN = "pugarch123"

model = None

# ✅ Model loader (lazy)
def get_model():
    global model
    if model is None:
        model = YOLO("best.pt")
    return model

# 🔒 Auth decorator
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token != f"Bearer {AUTH_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# 🎯 Prediction route with auth check
@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read uploaded image
    image_file = request.files['image']
    image_bytes = image_file.read()

    # Save to temp file to pass to model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    model = get_model()
    results = model(temp_file_path)

    # Plot result on original image
    plotted_img = results[0].plot()  # NumPy array

    # Convert NumPy to PIL Image
    image_pil = Image.fromarray(plotted_img)

    # Save to buffer
    img_io = io.BytesIO()
    image_pil.save(img_io, format='JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

# 🟢 Run the app
if __name__ == '__main__':
    app.run(debug=True)
