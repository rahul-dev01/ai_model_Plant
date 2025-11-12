from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model



import numpy as np
import cv2
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


model = load_model('plant_disease_model.h5')
CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust']

@app.route('/')
def upload_page():
    return render_template('upload.html') 

@app.route('/detection/upload/', methods=['POST'])
def predict():
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image.reshape(1, 256, 256, 3)

    prediction = model.predict(image)
    result = CLASS_NAMES[np.argmax(prediction)]
    return jsonify({'result': f"{result.split('-')[0]} leaf with {result.split('-')[1]}"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))  
    app.run(host="0.0.0.0", port=port)