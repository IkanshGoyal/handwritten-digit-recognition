import numpy as np
import mnist
import pickle
import function
import cv2

# Define the forward pass function
def forward_pass(input_image, weight1, bias1, weight2, bias2):
    input_layer = np.dot(input_image, weight1)
    hidden_layer = function.relu(input_layer + bias1)
    scores = np.dot(hidden_layer, weight2) + bias2
    probabilities = function.softmax(scores)
    return probabilities

# Load data
num_classes = 10
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Data processing
X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32')
x_test = X_test / 255
y_test = test_labels

# Load weights and biases
with open('weights.pkl', 'rb') as handle:
    b = pickle.load(handle, encoding="latin1")

weight1 = b[0]
bias1 = b[1]
weight2 = b[2]
bias2 = b[3]

# Make predictions for each test image
for num in range(test_images.shape[0]):
    input_image = x_test[num:num+1]
    predictions = forward_pass(input_image, weight1, bias1, weight2, bias2)
    predicted_class = np.argmax(predictions)

    print(f"Test Image {num+1}: Predicted Class = {predicted_class}")

    # Display the image with the predicted class
    img = np.zeros([28,28,3])
    img[:,:,0] = test_images[num]
    img[:,:,1] = test_images[num]
    img[:,:,2] = test_images[num]

    resized_image = cv2.resize(img, (100, 100))
    cv2.putText(resized_image, str(predicted_class), (5,20), cv2.FONT_HERSHEY_DUPLEX, .7, (0,255, 0), 1)
    cv2.imshow('input', resized_image)
    k = cv2.waitKey(0)
    if k == 27:
        break

cv2.destroyAllWindows()
