from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import base64
from src.hand_tracking import HandTracker

app = Flask(__name__)

# Load model once
model = joblib.load("models/sign_classifier.pkl")
CONFIDENCE_THRESHOLD = 0.8

# Initialize tracker once
tracker = HandTracker()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]

        # Decode base64 image
        image_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"prediction": "", "confidence": ""})

        # Get landmarks
        landmarks = tracker.get_landmarks(frame)

        if landmarks:
            input_data = np.array(landmarks).reshape(1, -1)

            probabilities = model.predict_proba(input_data)[0]
            confidence = float(np.max(probabilities))
            prediction = model.predict(input_data)[0]

            if confidence < CONFIDENCE_THRESHOLD:
                return jsonify({
                    "prediction": "",
                    "confidence": round(confidence, 2)
                })

            return jsonify({
                "prediction": prediction,
                "confidence": round(confidence, 2)
            })

        return jsonify({
            "prediction": "",
            "confidence": ""
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "prediction": "",
            "confidence": ""
        })


if __name__ == "__main__":
    app.run(debug=True)