import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Landmark indices for face direction
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
CHIN = 152
NOSE = 1

def get_face_direction(landmarks, w, h):
    left_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE], axis=0)
    right_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE], axis=0)
    nose = (landmarks[NOSE].x * w, landmarks[NOSE].y * h)
    chin = (landmarks[CHIN].x * w, landmarks[CHIN].y * h)

    eye_dx = right_eye[0] - left_eye[0]
    nose_dx = nose[0] - left_eye[0]
    horizontal_ratio = nose_dx / eye_dx

    vertical_ratio = (chin[1] - nose[1]) / (chin[1] - left_eye[1])

    if horizontal_ratio < 0.35:
        return "Left"
    elif horizontal_ratio > 0.65:
        return "Right"
    elif vertical_ratio > 0.80:
        return "Up"
    elif vertical_ratio < 0.45:
        return "Down"
    else:
        return "Center"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw mesh on face
        mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)

        # Get face direction
        face_dir = get_face_direction(landmarks, w, h)

        # Show direction on screen
        cv2.putText(frame, f"Face Direction: {face_dir}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Face Direction Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
