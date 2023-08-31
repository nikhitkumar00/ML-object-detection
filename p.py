import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mod.h5")

with open("list.txt", "r") as f:
    class_labels = f.read().splitlines()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)

    predictions = model.predict(input_data)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

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

cap.release()
cv2.destroyAllWindows()
