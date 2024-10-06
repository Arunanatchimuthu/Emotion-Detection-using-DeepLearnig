This project is an **Emotion Detection System** that identifies emotions from audio recordings using machine learning techniques. It leverages the RAVDESS dataset, which consists of various emotional speech recordings. The application uses a trained model to classify emotions based on audio features extracted from the input audio files.

app.py: The main Flask application file that sets up the server and defines the API endpoints. It handles audio file uploads, processes the files, and returns the predicted emotion.

RAVDESS.code-workspace: This file contains the workspace settings for Visual Studio Code, which helps manage the project environment.

RAVDESS.ipynb: A Jupyter Notebook that demonstrates the model training process, feature extraction, and evaluation metrics. This notebook includes detailed code and explanations for each step involved in building the emotion recognition model.


dataset:  https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

ravdess_emotion_recognition_model.h5: The saved Keras model file, which contains the trained model for emotion detection. This file is used by the Flask app to predict emotions from uploaded audio files.

/templates/index.html: The frontend HTML file that provides the user interface for the application. It includes an input field for uploading audio files and a button to detect the emotion. Inline CSS is used to style the webpage.
