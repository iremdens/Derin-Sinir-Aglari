from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("model.h5")
class_names = ['cataract', 'glaucoma', 'normal']

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert('RGB')
            img_array = preprocess(img)

            pred = model.predict(img_array)
            prediction = class_names[np.argmax(pred)]
            confidence = round(np.max(pred) * 100, 2)

            img_path = filepath

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
