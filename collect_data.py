# Import required libraries
import cv2  # OpenCV for video capture and image processing
import csv  # CSV module for writing data to files
from src.hand_tracking import HandTracker  # Custom module for hand landmark detection


def main():
    # Initialize the hand tracker object for detecting hand landmarks
    tracker = HandTracker()
    # Open the default webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    # Prompt user to enter a label (single letter A-Z) for the sign being recorded
    label = input("Enter label (A-Z): ").strip().upper()

    # Validate that the input is exactly one alphabetic character
    if len(label) != 1 or not label.isalpha():
        print("Invalid label. Please enter a single letter A–Z.")
        return

    # Open the CSV file in append mode to save hand landmark data
    with open("data/raw/sign_data.csv", "a", newline="") as f:
        writer = csv.writer(f)

        print("Recording data. Press 'q' to stop.")

        # Main loop to capture video frames and extract hand landmarks
        while True:
            # Read a frame from the video capture
            ret, frame = cap.read()
            # Exit loop if frame couldn't be read (end of video or camera disconnected)
            if not ret:
                break

            # Extract hand landmark coordinates from the current frame
            landmarks = tracker.get_landmarks(frame)

            # If landmarks were successfully detected, save them to CSV
            if landmarks:
                # Write landmark data along with the label to the CSV file
                writer.writerow(landmarks + [label])
                # Display "Captured" text on the frame to provide visual feedback
                cv2.putText(
                    frame,
                    "Captured",
                    (10, 40),  # Position: 10px from left, 40px from top
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font style
                    1,  # Font scale
                    (0, 255, 0),  # Color in BGR format (green)
                    2  # Line thickness
                )

            # Display the current frame in a window
            cv2.imshow("Data Collection", frame)

            # Check if 'q' key is pressed to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture resource
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Entry point: run the main function when the script is executed directly
if __name__ == "__main__":
    main()
