from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io

app = Flask(__name__)

# Load the pre-trained model
my_model = tf.keras.models.load_model('path_to_save_model.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image_data']
    
    # Convert base64 data to bytes
    image_bytes = base64.b64decode(image_data.split(',')[1])

    # Create a PIL Image from the bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    image = image.resize((28, 28))
    image_array = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255.0

    predictions = my_model.predict(image_array)
    predicted_class = np.argmax(predictions)

    return jsonify({'predicted_class': int(predicted_class)})

@app.route('/feedback', methods=['POST'])
def feedback():
    retrain_image_data = request.form['image_data']
    retrain_image_bytes = base64.b64decode(retrain_image_data.split(',')[1])
    retrain_image = Image.open(io.BytesIO(retrain_image_bytes)).convert('L')
    retrain_image = retrain_image.resize((28, 28))
    retrain_image_array = np.array(retrain_image).reshape((1, 28, 28, 1)).astype('float32') / 255.0

    predictions = my_model.predict(retrain_image_array)
    predicted_class = np.argmax(predictions)
        
    dynamic_train_model(my_model, retrain_image_array, predicted_class)

    return jsonify({'status': 'success'})

def dynamic_train_model(model, train_features, correct_label):
    correct_label_one_hot = tf.keras.utils.to_categorical(correct_label, num_classes=10)

    train_features_reshaped = train_features.reshape((1, 28, 28, 1)).astype('float32') / 255.0

    model.train_on_batch(train_features_reshaped, correct_label_one_hot)
    model.save('path_to_save_model.h5')

if __name__ == '__main__':
    app.run(debug=True)