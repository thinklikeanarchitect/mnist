from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Flask app nesnesi oluşturuluyor
app = Flask(__name__)

# Model yükleniyor
model = load_model('mnist_cnn_model.h5')

# Rota tanımlamaları
@app.route('/')
def home():
    return "Welcome to the MNIST prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image /= 255

    prediction = model.predict(image).argmax()
    return jsonify({'prediction': int(prediction)})

# Ana blok
if __name__ == '__main__':
    app.run(debug=True)
