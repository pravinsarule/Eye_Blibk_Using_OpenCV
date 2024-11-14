import streamlit as st
import cv2
import mediapipe as mp
from math import sqrt
import numpy as np

# Initialize Mediapipe and constants
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)
COUNTER = 0
TOTAL_BLINKS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Define landmarks for blink detection
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Functions for landmark detection and blink ratio
def landmarksDetection(image, results):
    image_height, image_width = image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    return mesh_coordinates

def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatio(image, landmarks, right_indices, left_indices):
    right_eye_h_distance = euclideanDistance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    right_eye_v_distance = euclideanDistance(landmarks[right_indices[12]], landmarks[right_indices[4]])
    left_eye_h_distance = euclideanDistance(landmarks[left_indices[0]], landmarks[left_indices[8]])
    left_eye_v_distance = euclideanDistance(landmarks[left_indices[12]], landmarks[left_indices[4]])
    right_eye_ratio = right_eye_h_distance / right_eye_v_distance
    left_eye_ratio = left_eye_h_distance / left_eye_v_distance
    return (right_eye_ratio + left_eye_ratio) / 2

# Streamlit app setup
st.title("Real-Time Eye Blink Detection")
st.write("**Instructions:** Blink your eyes to test the detection. The app will count your blinks.")

# Use Streamlit's camera input
camera_input = st.camera_input("Enable your camera")

# Display placeholder for blink counter
frame_placeholder = st.empty()
if camera_input:
    # Convert the captured image into a format OpenCV can process
    img = np.array(camera_input)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize and process frame
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Blink detection logic
    if results.multi_face_landmarks:
        mesh_coordinates = landmarksDetection(frame, results)
        eyes_ratio = blinkRatio(frame, mesh_coordinates, RIGHT_EYE, LEFT_EYE)
        cv2.putText(frame, "Please blink your eyes", (50, 100), FONT, 1, (0, 255, 0), 2)

        # Blink detection
        global COUNTER, TOTAL_BLINKS
        if eyes_ratio > 3:
            COUNTER += 1
        else:
            if COUNTER > 4:
                TOTAL_BLINKS += 1
                COUNTER = 0

        # Display the blink count
        cv2.rectangle(frame, (20, 120), (290, 160), (0, 0, 0), -1)
        cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONT, 1, (0, 255, 0), 2)

    # Convert the frame back to RGB for Streamlit and display it
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
else:
    st.warning("Please enable your camera to start blink detection.")
