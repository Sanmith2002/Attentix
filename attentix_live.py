import cv2
from datetime import datetime
import time
import pandas as pd

# Constants
detection_interval = 1.5  # seconds
max_window_width = 600    # resize window for better view

# Init variables
logs = []
last_check_time = time.time()
attention_score = 100
face_detected = False
total_checks = 0
present_checks = 0

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

print("📷 Attentix running... Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # Resize frame
    frame = cv2.resize(frame, (max_window_width, int(frame.shape[0] * max_window_width / frame.shape[1])))

    # Convert to gray for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run detection every 1.5s
    current_time = time.time()
    if current_time - last_check_time >= detection_interval:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        total_checks += 1
        if len(faces) > 0:
            face_detected = True
            present_checks += 1
            print("✅ Face Detected")
        else:
            face_detected = False
            print("❌ No Face")

        # Update attention score (percentage)
        attention_score = int((present_checks / total_checks) * 100)

        # Log entry
        logs.append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "face_present": 1 if face_detected else 0,
            "attention_score": attention_score
        })

        last_check_time = current_time

    # Draw face boxes if detected (re-detect for drawing, optional)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display detection status
    status_text = "✅ Face Detected" if face_detected else "❌ No Face"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0) if face_detected else (0, 0, 255), 2)

    # Draw attention score bar
    bar_width = int((attention_score / 100) * 200)
    cv2.rectangle(frame, (10, 50), (10 + 200, 80), (200, 200, 200), -1)  # full bar
    cv2.rectangle(frame, (10, 50), (10 + bar_width, 80), (0, 200, 0), -1)  # progress
    cv2.putText(frame, f"Attention: {attention_score}%", (220, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("🧠 Attentix Attention Monitor", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save logs
df = pd.DataFrame(logs)
df.to_csv("face_log.csv", index=False)

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Print final attention score
print(f"\n📝 Attention log saved as face_log.csv ✅")
print(f"📊 Final Attention Score: {attention_score}%")
