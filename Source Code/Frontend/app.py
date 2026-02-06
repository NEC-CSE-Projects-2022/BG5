import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from tensorflow import keras

# ---------------------------
# Load Segmentation Model (U-Net++)
# ---------------------------
segmentation_model = keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "unetpp_model2.keras"),
    compile=False
)

# ---------------------------
# Configs
# ---------------------------
THRESHOLD = 0.85  
MASK_TEMP_PATH = os.path.join("static", "mask_temp.jpg")
OVERLAY_TEMP_PATH = os.path.join("static", "overlay_temp.jpg")

# ---------------------------
# Detect if image is MASK image
# ---------------------------
def is_mask_image(img):
    unique_vals = np.unique(img)
    return set(unique_vals).issubset({0, 255})

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image_resized = image.resize((128, 128))
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = image_np[np.newaxis, ..., np.newaxis]
    return image_np

# ---------------------------
# Prediction Function (MODEL)
# ---------------------------
def predict_image(image_path, return_mask=False):
    image_np = preprocess_image(image_path)
    mask = segmentation_model.predict(image_np)

    mask_bin = (mask > THRESHOLD).astype(np.float32)
    tumor_area = np.sum(mask_bin)
    max_value = mask.max()

    if return_mask:
        mask_img = Image.fromarray((mask_bin[0, :, :, 0] * 255).astype(np.uint8))
        mask_img.save(MASK_TEMP_PATH)

    MIN_TUMOR_AREA = 50
    MIN_MASK_VALUE = 0.5

    if tumor_area < MIN_TUMOR_AREA or max_value < MIN_MASK_VALUE:
        result = {
            "prediction": "Normal(No Tumor)",
            "is_tumor": False,
            "description": "No abnormal region detected.",
            "recommendation": "No further action needed."
        }
    else:
        result = {
            "prediction": "Tumor Detected",
            "is_tumor": True,
            "description": "Abnormal region detected by segmentation.",
            "recommendation": "Consult a medical expert for further analysis."
        }

    if return_mask:
        result["mask_image"] = MASK_TEMP_PATH

    return result

# ---------------------------
# Overlay Function
# ---------------------------
def create_overlay(original_image_path, mask_path, output_path=OVERLAY_TEMP_PATH):
    try:
        orig = cv2.imread(original_image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if orig is None or mask is None:
            return None

        mask_resized = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
        contours, _ = cv2.findContours((mask_resized > 128).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        outlined = orig.copy()
        cv2.drawContours(outlined, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(output_path, outlined)
        return output_path
    except:
        return None

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def index():
    return redirect("/home")

@app.route("/home")
def home():
    return render_template("Home.html")

@app.route("/features")
def features_page():
    return render_template("Features.html")

@app.route("/upload")
def upload_page():
    return render_template("Upload.html")

@app.route("/login")
def login_page():
    return render_template("Login.html")

@app.route("/contact")
def contact():
    return render_template("Contact.html")

@app.route("/signup")
def signup():
    return render_template("Signup.html")

@app.route('/feature_high_accuracy')
def feature_high_accuracy():
    return render_template('feature_high_accuracy.html')

@app.route('/feature_fast_processing')
def feature_fast_processing():
    return render_template('feature_fast_processing.html')

@app.route('/feature_detailed_insights')
def feature_detailed_insights():
    return render_template('feature_detailed_insights.html')

@app.route("/awareness")
def awareness():
    return render_template("awareness.html")

@app.route("/treatments")
def treatments():
    return render_template("treatments.html")

@app.route("/statistics")
def statistics():
    return render_template("statistics.html")

# --------------------------------------------------
# UPDATED /predict — supports MASK + ULTRASOUND
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = os.path.join("static", "temp.jpg")
    file.save(temp_path)

    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

    # -------- CASE 1: MASK IMAGE ----------
    if is_mask_image(img):
        print("Mask image detected — skipping model.")

        white_pixels = np.sum(img == 255)

        # save mask as mask_temp.jpg for overlay
        cv2.imwrite(MASK_TEMP_PATH, img)

        if white_pixels == 0:
            result = {
                "prediction": "Normal(No Tumor)",
                "is_tumor": False,
                "description": "Mask has no white region.",
                "recommendation": "No tumor region found."
            }
        else:
            result = {
                "prediction": "Tumor Detected",
                "is_tumor": True,
                "description": "Tumor region detected in mask.",
                "recommendation": "Consult a medical expert."
            }

        # create outline
        overlay_path = create_overlay(temp_path, MASK_TEMP_PATH)
        result["overlay_image"] = "/" + overlay_path.replace("\\", "/")

        return jsonify(result)

    # -------- CASE 2: ULTRASOUND IMAGE ----------
    result = predict_image(temp_path, return_mask=True)

    overlay_path = create_overlay(temp_path, MASK_TEMP_PATH)
    if overlay_path:
        result["overlay_image"] = "/" + overlay_path.replace("\\", "/")

    return jsonify(result)

# ---------------------------
# View Details
# ---------------------------
@app.route("/details")
def view_details():
    if os.path.exists(OVERLAY_TEMP_PATH):
        return render_template("Details.html", image_path="/" + OVERLAY_TEMP_PATH.replace("\\", "/"))
    else:
        return "No highlighted tumor image found."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
