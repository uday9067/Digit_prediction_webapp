import numpy as np
import cv2
from flask import Flask, render_template, request
import tensorflow as tf
import os

app = Flask(__name__, template_folder='templets')

# Load trained model
model = tf.keras.models.load_model('digit_model.h5')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('file.html')

from werkzeug.utils import secure_filename

@app.route('/file', methods=['GET', 'POST'])
def file():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('file.html', prediction="No file uploaded", file=None)

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Read as grayscale
        image = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return render_template('file.html', prediction="Invalid image file", file=None)

        # Resize if not already 28x28
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28))

        # Normalize and reshape
        image = image.astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)

        try:
            pred_array = model.predict(image)
            prediction = int(np.argmax(pred_array))
        except Exception as e:
            return render_template('file.html', prediction=f"Prediction error: {str(e)}", file=filename)

    return render_template('file.html', prediction=prediction, file=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
