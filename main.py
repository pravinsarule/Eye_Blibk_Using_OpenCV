import streamlit as st
import cv2
import mediapipe
from math import sqrt
import numpy as np

__author__ = "Shakir Sadiq"

# Constants and setup
COUNTER = 0
TOTAL_BLINKS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# landmarks from mesh_map.jpg
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mediapipe_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mediapipe_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)

# Function to detect landmarks
def landmarksDetection(image, results, draw=False):
    image_height, image_width = image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(image, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
    return mesh_coordinates

# Function for Euclidean distance calculation
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blink Ratio function
def blinkRatio(image, landmarks, right_indices, left_indices):
    right_eye_landmark1 = landmarks[right_indices[0]]
    right_eye_landmark2 = landmarks[right_indices[8]]
    right_eye_landmark3 = landmarks[right_indices[12]]
    right_eye_landmark4 = landmarks[right_indices[4]]
    left_eye_landmark1 = landmarks[left_indices[0]]
    left_eye_landmark2 = landmarks[left_indices[8]]
    left_eye_landmark3 = landmarks[left_indices[12]]
    left_eye_landmark4 = landmarks[left_indices[4]]

    right_eye_horizontal_distance = euclaideanDistance(right_eye_landmark1, right_eye_landmark2)
    right_eye_vertical_distance = euclaideanDistance(right_eye_landmark3, right_eye_landmark4)
    left_eye_vertical_distance = euclaideanDistance(left_eye_landmark3, left_eye_landmark4)
    left_eye_horizobtal_distance = euclaideanDistance(left_eye_landmark1, left_eye_landmark2)

    right_eye_ratio = right_eye_horizontal_distance / right_eye_vertical_distance
    left_eye_ratio = left_eye_horizobtal_distance / left_eye_vertical_distance

    eyes_ratio = (right_eye_ratio + left_eye_ratio) / 2
    return eyes_ratio

# Streamlit app interface
st.title("Real-Time Eye Blink Detection")

# Create a video capture object and initialize variables
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    st.error("Cannot access the camera. Please check your camera permissions.")
    exit()


# Display blink counter and instructions
st.write("**Instructions:**")
st.write("1. Blink your eyes to test the detection.")
st.write("2. The app will count your blinks.")

# Placeholder for displaying video frames in Streamlit
frame_placeholder = st.empty()

# Main loop
while True:
    ret, frame = video_capture.read()

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coordinates = landmarksDetection(frame, results, True)

        eyes_ratio = blinkRatio(frame, mesh_coordinates, RIGHT_EYE, LEFT_EYE)

        cv2.putText(frame, "Please blink your eyes", (int(frame_height/2), 100), FONT, 1, (0, 255, 0), 2)

        if eyes_ratio > 3:
            COUNTER += 1
        else:
            if COUNTER > 4:
                TOTAL_BLINKS += 1
                COUNTER = 0

        # Draw total blinks on the frame
        cv2.rectangle(frame, (20, 120), (290, 160), (0, 0, 0), -1)
        cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONT, 1, (0, 255, 0), 2)

    # Display the frame in the Streamlit app using the new parameter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)


    # Check if ESC key is pressed to break the loop
    if cv2.waitKey(1) == 27:
        break

# Release resources when done
video_capture.release()
cv2.destroyAllWindows()
