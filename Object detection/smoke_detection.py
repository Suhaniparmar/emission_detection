import cv2
import numpy as np

# Function to detect smoke in a frame
def detect_smoke(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve detection
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smoke_detected = False
    
    for contour in contours:
        # If contour area is large enough, consider it as smoke
        if cv2.contourArea(contour) > 5000:
            smoke_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return frame, smoke_detected

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect smoke in the frame
    frame, smoke_detected = detect_smoke(frame)
    
    # Display the resulting frame
    cv2.imshow('Smoke Detection', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
