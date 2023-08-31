# Real-time Object Detection using Keras and OpenCV

This repository contains a simple Python script for real-time object detection using a pre-trained Keras model and OpenCV. The script captures video from the default camera (usually the webcam), processes each frame using the loaded Keras model, and displays the frame with the predicted object class label.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- TensorFlow (`tensorflow`)
- A trained Keras model saved in `keras_model.h5`
- A text file `labels.txt` containing the class labels corresponding to the model's output

You can install the required Python packages using the following command:

```
pip install opencv-python numpy tensorflow
```

## Usage

1. Place your trained Keras model (`keras_model.h5`) and a text file with class labels (`labels.txt`) in the same directory as the script.

2. Update the `keras_model.h5` and `labels.txt` filenames in the script if they're named differently.

3. Run the script using the following command:

```
python object_detection.py
```

4. A window will appear displaying the real-time video stream from your camera. The predicted class label for the detected object will be displayed on each frame.

5. To exit the script, press the 'q' key in the OpenCV window.

## Notes

- The script assumes that your Keras model takes input images of size `(224, 224, 3)` and outputs class probabilities.
- The camera feed can be adjusted by modifying the `cap = cv2.VideoCapture(0)` line. Replace `0` with the index of your desired camera if you have multiple cameras connected.
- Make sure your camera is accessible and working properly before running the script.
- This is a basic example and might need modifications to fit specific use cases or models.

## Credits

This script is inspired by tutorials and examples from the OpenCV and TensorFlow communities.

---