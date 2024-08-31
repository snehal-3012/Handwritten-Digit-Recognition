import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("handwritten_digit_recognition_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (28, 28))
    image_normalized = image_resized.astype('float32') / 255.0
    image_inverted = 1 - image_normalized 
    image_reshaped = np.reshape(image_inverted, (1, 28, 28, 1))
    return image_reshaped

# Function to make predictions on the preprocessed image
def predict_digit(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    print(f"Predicted digit: {predicted_digit}")
    plt.imshow(preprocessed_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()

image_path = '/test_img.png'  # Replace with your image file path
predict_digit(image_path)
