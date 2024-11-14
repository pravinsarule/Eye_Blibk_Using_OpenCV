# Streamlit app interface
import streamlit as st
import cv2
import mediapipe
from math import sqrt

# Initialize variables
COUNTER = 0
TOTAL_BLINKS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Initialize face mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mediapipe_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mediapipe_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)

# Initialize webcam capture
video_capture = cv2.VideoCapture(0)

# Ensure the camera is accessible
if not video_capture.isOpened():
    st.error("Cannot access the camera. Please check your camera permissions.")
    exit()

# Display blink counter and instructions
st.title("Real-Time Eye Blink Detection")
st.write("**Instructions:** Blink your eyes to test the detection.")

frame_placeholder = st.empty()

while True:
    ret, frame = video_capture.read()

    # Check if frame is valid
    if not ret or frame is None:
        st.error("Failed to capture image from webcam.")
        break

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coordinates = landmarksDetection(frame, results, True)
        eyes_ratio = blinkRatio(frame, mesh_coordinates, RIGHT_EYE, LEFT_EYE)

        cv2.putText(frame, "Please blink your eyes", (int(frame_height / 2), 100), FONT, 1, (0, 255, 0), 2)

        if eyes_ratio > 3:
            COUNTER += 1
        else:
            if COUNTER > 4:
                TOTAL_BLINKS += 1
                COUNTER = 0

        # Draw total blinks on the frame
        cv2.rectangle(frame, (20, 120), (290, 160), (0, 0, 0), -1)
        cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONT, 1, (0, 255, 0), 2)

    # Convert frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame in Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Break the loop if ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
