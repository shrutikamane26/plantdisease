from flask import Flask, request, jsonify, render_template
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model(r"D:\sem 8\ML\Coconut Tree Disease Data - Copy\CoconutDisease.h5")

# Labels
labels = {0: "Bud Root Dropping", 1: "Bud Rot", 2: "Gray Leaf Spot", 3: "Leaf Rot", 4: "Stem Bleeding"}

# Image processing function
def process_img(img_path):
    img = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    res = labels.get(y, "Unknown")
    accuracy = np.max(answer) * 100
    return res.capitalize(), accuracy

# Serve images from upload_images directory
@app.route('/upload_images/<filename>')
def uploaded_file(filename):
    return send_from_directory('upload_images', filename)

# Routes
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Save the file
        file_path = os.path.join('upload_images', file.filename)
        file.save(file_path)
        
        # Process the image and make a prediction
        result, accuracy = process_img(file_path)
        
        # Return the file path to be used in the HTML response
        return jsonify({
            'category': result,
            'accuracy': f'{accuracy:.2f}%',
            'image_path': f'/upload_images/{file.filename}'
        })

if __name__ == '__main__':
    # Ensure the upload_images directory exists
    if not os.path.exists('upload_images'):
        os.makedirs('upload_images')
    app.run(debug=True)