from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # CORS'u etkinleştirin
model = load_model('mnist_cnn_model.h5')

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

if __name__ == '__main__':
    app.run(debug=True)
