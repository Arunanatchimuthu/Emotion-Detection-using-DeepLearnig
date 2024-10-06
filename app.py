from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('ravdess_emotion_recognition_model.h5')

# Define the emotions
EMOTIONS = ['Neutral😐', 'Calm🙍🏾‍♀️', 'Happy😊!!', 'Sad😔', 'Angry😡', 'Fearful😨😬', 'Disgust', 'Surprised🙀!!']

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to extract MFCC features
def extract_features(audio_path):
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    audio = request.files['audio']
    if audio.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(audio.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio.save(audio_path)

    # Extract features and predict
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    predicted_emotion = np.argmax(prediction)

    emotion_label = EMOTIONS[predicted_emotion]

    return jsonify({'emotion': emotion_label})

if __name__ == '__main__':
    app.run(debug=True)
