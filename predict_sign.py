import cv2
import joblib
import numpy as np
from collections import deque
from src.hand_tracking import HandTracker

# Load trained model
model = joblib.load("models/sign_classifier.pkl")

# Initialize hand tracker
tracker = HandTracker()

# Start webcam
cap = cv2.VideoCapture(0)

# Store last predictions for smoothing
prediction_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame (mirror effect for natural interaction)
    frame = cv2.flip(frame, 1)

    # Get landmarks
    landmarks = tracker.get_landmarks(frame)

    if landmarks:
        # Convert to numpy array
        data = np.array(landmarks).reshape(1, -1)

        # Predict
        prediction = model.predict(data)[0]

        # Add prediction to buffer
        prediction_buffer.append(prediction)

        # Get most frequent prediction
        final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

        # Display prediction
        cv2.putText(
            frame,
            f"Prediction: {final_prediction}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        # Clear buffer if no hand detected
        prediction_buffer.clear()

    cv2.imshow("Sign Prediction", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit if window is closed
    if cv2.getWindowProperty("Sign Prediction", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()