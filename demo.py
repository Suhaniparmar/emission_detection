import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('C:\\Users\\Suhani\\Desktop\\Python\\Object detection\\pretrained_mobilenetv2.h5')

# Function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict using the model
    prediction = model.predict(preprocessed_frame)

    # Check the shape of prediction
    if prediction.shape[-1] > 1:  # Assuming prediction is a 1D array of probabilities
        # Use a threshold to determine if smoke is detected
        smoke_detected = np.any(prediction > 0.5)  # Adjust threshold as necessary
    else:
        smoke_detected = prediction[0][0] > 0.5  # Adjust this based on the structure of your prediction

    # Display result
    label = 'Smoke Detected' if smoke_detected else 'No Smoke'
    color = (0, 0, 255) if smoke_detected else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Smoke Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
