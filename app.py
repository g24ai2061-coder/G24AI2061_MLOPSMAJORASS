# app.py
import joblib
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image

# Load model package at startup (dict with model + scaler)
model_package = joblib.load("savedmodel.pth")
model = model_package["model"]
scaler = model_package.get("scaler", None)

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h1>Upload a face image (will be resized to 64x64)</h1>
<form method="POST" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <input type="submit" value="Predict">
</form>

{% if prediction is not none %}
  <h2>Predicted Class: {{ prediction }}</h2>
{% endif %}
"""


def preprocess_image(file_storage):
    # Read image bytes -> grayscale -> resize -> flatten -> [0,1]
    img = Image.open(file_storage.stream).convert("L").resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.flatten().reshape(1, -1)
    return arr


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            x = preprocess_image(file)

            # Apply same scaler used during training
            if scaler is not None:
                x = scaler.transform(x)

            # Predict with the trained DecisionTree model
            pred = model.predict(x)[0]
            prediction = int(pred)

    return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=4000, debug=True)