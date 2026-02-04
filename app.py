from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os


# ✅ Auto Download Model If Not Present
if not os.path.exists("leaf_model.h5"):
    import download_model

# ✅ Load Model
model = tf.keras.models.load_model("leaf_model_resnet50_final.h5")
print("✅ Model Loaded Successfully!")


app = Flask(__name__)

# ✅ Load Model
model = tf.keras.models.load_model("leaf_disease_resnet50_final.h5")

# ✅ Load Class Labels
with open("class_labels.json") as f:
    class_labels = json.load(f)

# Reverse mapping index → class name
labels = {v: k for k, v in class_labels.items()}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["leaf"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ✅ Preprocess Image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)

    # ✅ Prediction
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    return render_template(
        "result.html",
        prediction=predicted_class,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)