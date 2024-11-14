import streamlit as st
import cv2
import mediapipe as mp
from math import sqrt
import numpy as np

# Constants and setup
COUNTER = 0
TOTAL_BLINKS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# landmarks for eye regions from mediapipe
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)

# Function to detect landmarks
def landmarks_detection(image, results, draw=False):
    image_height, image_width = image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(image, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
    return mesh_coordinates

# Function for Euclidean distance calculation
def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blink ratio calculation
def blink_ratio(image, landmarks, right_indices, left_indices):
    right_eye_horizontal = euclidean_distance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    right_eye_vertical = euclidean_distance(landmarks[right_indices[12]], landmarks[right_indices[4]])
    left_eye_horizontal = euclidean_distance(landmarks[left_indices[0]], landmarks[left_indices[8]])
    left_eye_vertical = euclidean_distance(landmarks[left_indices[12]], landmarks[left_indices[4]])

    right_eye_ratio = right_eye_horizontal / right_eye_vertical
    left_eye_ratio = left_eye_horizontal / left_eye_vertical

    eyes_ratio = (right_eye_ratio + left_eye_ratio) / 2
    return eyes_ratio

# Streamlit app interface
st.title("Real-Time Eye Blink Detection")
st.write("**Instructions:** Blink your eyes to test the detection. The app will count your blinks.")

# Streamlit's camera input widget
camera_input = st.camera_input("Take a picture or start live video")

# Placeholder for displaying video frames in Streamlit
frame_placeholder = st.empty()

if camera_input:
    # Convert the Streamlit camera input to an OpenCV-compatible format (NumPy array)
    frame = np.array(camera_input)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV processing

    # Process the frame with MediaPipe
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        mesh_coordinates = landmarks_detection(frame, results, True)
        eyes_ratio = blink_ratio(frame, mesh_coordinates, RIGHT_EYE, LEFT_EYE)

        # Display instructions on the frame
        cv2.putText(frame, "Please blink your eyes", (50, 50), FONT, 1, (0, 255, 0), 2)

        # Blink detection logic
        if eyes_ratio > 3:
            COUNTER += 1
        else:
            if COUNTER > 4:
                TOTAL_BLINKS += 1
                COUNTER = 0

        # Draw blink count on the frame
        cv2.rectangle(frame, (20, 120), (300, 160), (0, 0, 0), -1)
        cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONT, 1, (0, 255, 0), 2)

    # Convert frame back to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

else:
    st.warning("Please enable your camera and refresh the page if necessary.")
