import cv2
import torch
import face_recognition
import numpy as np
import os
import smtplib
import time
from pathlib import Path

# Load YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to the folder containing images of authorized faces
AUTHORIZED_DIR = 'authorized_faces'

# Load authorized faces and encode them
authorized_faces_encodings = []
authorized_faces_names = []

for image_name in os.listdir(AUTHORIZED_DIR):
    image_path = os.path.join(AUTHORIZED_DIR, image_name)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    authorized_faces_encodings.append(encoding)
    authorized_faces_names.append(os.path.splitext(image_name)[0])  # Use the filename as the person's name

# Function to send email alert
def send_email_alert(unauthorized_person):
    try:
        # Set up the server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("vaishrao734@gmail.com", "ioyx mxwh nasa jwag")  # Replace with your email and password
        
        # Create the email content
        subject = "ALERT: Unauthorized Person Detected"
        body = f"Unauthorized person ({unauthorized_person}) has been detected!"
        message = f"Subject: {subject}\n\n{body}"
        
        # Send the email
        server.sendmail("vaishrao734@gmail.com", "vaishnavi7304@gmail.com", message)
        server.quit()
        
        print(f"Email alert sent for {unauthorized_person}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Initialize video capture (0 for webcam or specify a video file path)
video_source = 0
cap = cv2.VideoCapture(video_source)

last_alert_time = {}
COOLDOWN_PERIOD = 60

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO inference for person detection
    results = model(frame)

    # Loop through detected objects
    for obj in results.xyxy[0]:  # xyxy is the bounding box format
        class_id = int(obj[5])
        label = model.names[class_id]
        confidence = obj[4]
        if confidence > 0.5:  # Only consider detections above 50% confidence
            x1, y1, x2, y2 = map(int, obj[:4])
            if label == 'person':  # Only process people
                x1, y1, x2, y2 = map(int, obj[:4])  # Extract bounding box coordinates
                # Assuming this is inside the main loop where you are processing frames

# Handle face recognition for detected persons
                face_frame = frame[y1:y2, x1:x2]
                if face_frame.shape[2] != 3 or face_frame.dtype != np.uint8:
                    print("Invalid face frame shape or type.")
                    continue
# Check if the cropped frame has the expected dimensions
                if face_frame.shape[0] == 0 or face_frame.shape[1] == 0:
                    continue  # Skip this iteration if the face frame is empty

                face_locations = face_recognition.face_locations(face_frame)

# Get face encodings; this function should handle empty detections
                face_encodings = face_recognition.face_encodings(face_frame, face_locations)

                if len(face_encodings) == 0:
                    continue  # Skip if no faces found

# Process each detected face
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(authorized_faces_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(authorized_faces_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        person_name = authorized_faces_names[best_match_index]
                        cv2.putText(frame, f"Authorized: {person_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        person_name = "Unauthorized Person"
                        cv2.putText(frame, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        current_time = time.time()
                        if person_name not in last_alert_time or (current_time - last_alert_time[person_name]) > COOLDOWN_PERIOD:
                            send_email_alert(person_name)
                            last_alert_time[person_name] = current_time

            # Draw bounding box around the detected person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
                # Draw bounding box around detected objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for non-person objects
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
     

    # Display the video frame with detections
    cv2.imshow("Surveillance Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
