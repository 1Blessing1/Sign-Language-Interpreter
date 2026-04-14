import cv2
import joblib
import numpy as np
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
from src.hand_tracking import HandTracker

# Load model
model = joblib.load("models/sign_classifier.pkl")

# Initialize tracker
tracker = HandTracker()

# Prediction buffer for smoothing
buffer = deque(maxlen=10)

# Store outputs
current_signs = []
sentence = ""

# Setup window
root = tk.Tk()
root.title("Sign Language Interpreter")

# Webcam display
video_label = tk.Label(root)
video_label.pack()

# Sentence bar
sentence_label = tk.Label(root, text="Sentence: ", font=("Arial", 16))
sentence_label.pack(fill="x")

# Signs bar
signs_label = tk.Label(root, text="Signs: ", font=("Arial", 14))
signs_label.pack(fill="x")

cap = cv2.VideoCapture(0)

def update_frame():
    global sentence, current_signs

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)

    landmarks = tracker.get_landmarks(frame)

    if landmarks:
        data = np.array(landmarks).reshape(1, -1)
        pred = model.predict(data)[0]

        buffer.append(pred)

        # Smooth prediction
        final_pred = max(set(buffer), key=buffer.count)

        # Add to signs (avoid duplicates spam)
        if len(current_signs) == 0 or current_signs[-1] != final_pred:
            current_signs.append(final_pred)

        # Limit size
        if len(current_signs) > 10:
            current_signs.pop(0)

        # Update UI
        signs_label.config(text=f"Signs: {' '.join(current_signs)}")

    # Show webcam
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Add buttons
def add_to_sentence():
    global sentence, current_signs
    if current_signs:
        sentence += current_signs[-1]
        sentence_label.config(text=f"Sentence: {sentence}")

def clear_all():
    global sentence, current_signs
    sentence = ""
    current_signs = []
    sentence_label.config(text="Sentence: ")
    signs_label.config(text="Signs: ")

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Add Letter", command=add_to_sentence).pack(side="left")
tk.Button(btn_frame, text="Clear", command=clear_all).pack(side="left")

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()