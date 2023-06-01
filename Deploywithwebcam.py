import cv2
import tensorflow as tf
import numpy as np

# Load the model architecture from JSON
model_architecture_path = 'image_classifier_model_5.json'
with open(model_architecture_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model weights
model_weights_path = 'image_classifier_weights_3.h5'

# Load the class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  
class_labels.sort()

# Load the model
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights(model_weights_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Preprocess the frame for prediction
    frame = cv2.resize(frame, (150, 150))
    input_data = np.expand_dims(frame, axis=0)
    input_data = input_data / 255.0

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)
    predicted_class = np.argmax(predictions[0])
    class_label = class_labels[predicted_class]

    # Display the predicted class label on the frame
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
