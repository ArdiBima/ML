import tensorflow as tf
import numpy as np

# Load the model architecture from JSON
model_architecture_path = 'image_classifier_model_5.json'
with open(model_architecture_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model weights
model_weights_path = 'image_classifier_weights_3.h5'

# Load the image to make predictions on
image_path = 'messageImage_1685453464935.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
input_data = tf.keras.preprocessing.image.img_to_array(image)
input_data = np.expand_dims(input_data, axis=0)
input_data /= 255.0

# Load the model
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights(model_weights_path)

# Make predictions using the loaded model
predictions = loaded_model.predict(input_data)
predicted_class = np.argmax(predictions[0])

# Define the class labels alphabetically
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  

# Print the predicted class label
print('Predicted class:', class_labels[predicted_class])
