import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('pneumoniaCovidTester.keras')

# Load and preprocess the image you want to classify
img_path = 'TestXRay.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(256, 256))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)  # Add batch dimension
img = img / 255.0  # Normalize pixel values

# Make predictions
predictions = model.predict(img)

# Get the predicted class (index with the highest probability)
predicted_class = np.argmax(predictions)

# Print the predicted class or label
if predicted_class == 0:
    print("Predicted: COVID-19")
elif predicted_class == 1:
    print("Predicted: Pneumonia")
elif predicted_class == 2:
    print("Predicted: Normal")
