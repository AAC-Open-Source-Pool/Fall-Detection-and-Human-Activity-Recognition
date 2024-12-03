import cv2
import cvzone
import math
from ultralytics import YOLO
from twilio.rest import Client
import time

# Twilio credentials
account_sid = 'AC9c2792485155513617056600d5544698'  # Replace with your Twilio Account SID
auth_token = '86bb97b2dcb204599c340979214c58d4'   # Replace with your Twilio Auth Token
twilio_number = +17754179012  # Replace with your Twilio phone number
your_number = +918341606749   # Replace with the number to receive SMS

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Initialize video capture
cap = cv2.VideoCapture('fall.mp4')
time.sleep(2)

if not cap.isOpened():
    print("Cannot open video or webcam")
    exit()

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
file = open('classes.txt', 'r')
for line in file:
    classnames.append(line.strip())
file.close()

# Position and angle tracking
position_history = {}
angle_history = {}

# Thresholds for fall detection
velocity_threshold = 20  
angle_change_threshold = 45  
aspect_ratio_threshold = 1.5  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame or end of video")
        break

    frame = cv2.resize(frame, (640, 640))  

    # Perform YOLO detection
    results = model(frame)

    for i in results:
        parameters = i.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            aspect_ratio = width / height
            center_y = (y1 + y2) // 2

            angle = math.degrees(math.atan2(height, width))

            # Generate a unique ID for tracking
            person_id = f'{x1}{y1}{x2}_{y2}'

            if person_id not in position_history:
                position_history[person_id] = []  
            if person_id not in angle_history:
                angle_history[person_id] = []  

            position_history[person_id].append(center_y)
            angle_history[person_id].append(angle)

            # Keep only the last two values for velocity and angle calculations
            if len(position_history[person_id]) > 2:
                position_history[person_id] = position_history[person_id][-2:]
            if len(angle_history[person_id]) > 2:
                angle_history[person_id] = angle_history[person_id][-2:]

            # Calculate velocity and angle change
            velocity = position_history[person_id][-1] - position_history[person_id][-2] if len(position_history[person_id]) >= 2 else 0
            angle_change = abs(angle_history[person_id][-1] - angle_history[person_id][-2]) if len(angle_history[person_id]) >= 2 else 0

            if conf > 80 and class_detect == 'person':
                threshold = height - width
                if aspect_ratio > aspect_ratio_threshold or velocity > velocity_threshold or angle_change > angle_change_threshold or threshold < 0:
                    fall_text_position = (x1 + width // 2 - 60, y1 - 45)
                    cvzone.putTextRect(frame, 'Fall Detected', fall_text_position, scale=1.5, thickness=2, offset=10, colorR=(0, 0, 255))

                    # Send SMS using Twilio
                    try:
                        message = client.messages.create(
                            body="Fall Detected! Immediate action may be required.",
                            from_=twilio_number,
                            to=your_number
                        )
                        print(f"Message sent: {message.sid}")
                    except Exception as e:
                        print(f"Failed to send SMS: {e}")

                # Draw rectangle and label around the detected person
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6, colorR=(0, 255, 0), colorC=(255, 0, 0))
                cvzone.putTextRect(frame, f'{class_detect} {conf}%', (x1, y1 - 15), scale=1.5, thickness=2, offset=10)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
