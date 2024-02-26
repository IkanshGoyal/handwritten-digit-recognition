from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import io
import numpy as np
import mnist
import pickle
import function

app = Flask(__name__)

def forward_pass(input_image, weight1, bias1, weight2, bias2):
    # Flatten the input image
    input_image_flattened = input_image.reshape((1, -1))

    input_layer = np.dot(input_image_flattened, weight1)
    hidden_layer = function.relu(input_layer + bias1)
    scores = np.dot(hidden_layer, weight2) + bias2
    probabilities = function.softmax(scores)
    return probabilities

# Load weights and biases
with open('weights.pkl', 'rb') as handle:
    b = pickle.load(handle, encoding="latin1")

weight1 = b[0]
bias1 = b[1]
weight2 = b[2]
bias2 = b[3]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.form['image_data'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255.0

        predictions = forward_pass(image_array, weight1, bias1, weight2, bias2)
        predicted_class = np.argmax(predictions)

        print("Predictions:", predictions, predicted_class)

        return jsonify({'predicted_class': int(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
