import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("keras_model.h5")

# Load class labels
with open("labels.txt", "r") as f:
    class_labels = f.read().splitlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess the frame for prediction
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)

    # Make prediction
    predictions = model.predict(input_data)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Draw bounding box and label
    cv2.putText(
        frame,
        predicted_class_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
