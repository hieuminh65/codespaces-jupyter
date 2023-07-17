#necessary imports
import librosa
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pickle import load
import sys
from flask import Flask, jsonify
from flask import Flask, jsonify, render_template, request, redirect, session
import json
import os
import onnxruntime
from config import Config
from pydub import AudioSegment
import base64   
from tempfile import NamedTemporaryFile



app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
@app.route('/process_data', methods=['POST', 'GET'])
def process_data():
    data = request.get_json()
    if 'audio' not in data:
        return jsonify({'error': 'Base64 audio data not found in the request.'}), 400
    
    audio = data['audio']
    
    # Decode the Base64 audio string
    decoded_audio = base64.b64decode(audio)
      
    try:
        # Save the file with a temporary name

        with NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
            temp_path = temp_file.name
            temp_file.write(decoded_audio)
        
        # Convert the temporary .m4a file to WAV format using pydub
       
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'],'temp.wav' )
        if os.path.exists(wav_path):
            os.remove(wav_path)
        audio = AudioSegment.from_file(temp_path, format='m4a')
        audio.export(wav_path, format='wav')

        return str(get_test_value(wav_path))

    except:
        return jsonify({'error': 'Failed to convert audio file to WAV format.'}), 400

    


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    print('working')
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    return result

def get_test_value(path):
    # model = load(open('model.pkl', 'rb'))
    session = onnxruntime.InferenceSession('model.onnx')

    scaler = load(open('scaler.pkl', 'rb'))
    encoder = load(open('encoder.pkl', 'rb'))
    x_test = get_features(path)
    x_test = np.array(x_test)
    x_test = np.expand_dims(x_test, axis=0)   

    x_test = scaler.transform(x_test)

    x_test = x_test.astype(np.float32)
    x_test = np.reshape(x_test, (1, x_test.shape[1], 1))

    #pred_test = model.predict(x_test)
    pred_test = session.run(None, {'conv1d_input': x_test})
    pred_test = np.expand_dims(pred_test, axis=0)  
    print('pred') 
    #print(pred_test[0][0]) 
    y_pred = encoder.inverse_transform(pred_test[0][0])

    print(y_pred[0][0])
    
    return y_pred[0][0]

def main():
    # Check if the user has provided the input path argument
    print('working')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    main()