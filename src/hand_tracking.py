# OpenCV for image processing and video capture
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np


class HandTracker:
    """
    A class to detect and track hand landmarks using MediaPipe.
    Provides hand pose estimation for video frames.
    """

    def __init__(self):
        """
        Initialize the hand tracker with MediaPipe's pre-trained hand detection model.
        """
        # Create hand landmarker using the new MediaPipe Tasks API
        base_options = mp.tasks.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def get_landmarks(self, frame):
        """
        Detect hand landmarks in a video frame.

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            List of 42 values (x, y for 21 landmarks) or None
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hand landmarks using the new Tasks API
        detection_result = self.hand_landmarker.detect(mp_image)

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks = []

            for lm in hand_landmarks:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            return landmarks

        return None
