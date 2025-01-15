from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


model_path = 'sizepred_model.h5'
model = load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_size(image_path):
    img_array = preprocess_image(image_path)
    img_array /= 255.0  
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    size_labels = ['Large', 'Medium', 'Small', 'X Large']
    predicted_size = size_labels[predicted_class]
    return predicted_size


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
    
        
        if file.filename == '':
            return render_template('index.html', message='No image selected')
        
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('index.html', message='Unsupported file type')
    
        
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
    
        
        predicted_size = predict_size(file_path)
       
       
        return render_template('result.html', filename=file.filename, size=predicted_size)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
