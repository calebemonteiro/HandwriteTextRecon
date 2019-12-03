import cv2
import sys
import base64
sys.path.append('src')
#
import numpy as np
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
#
from flask import Flask, request, jsonify
#
app = Flask(__name__)
#
#https://modeldepot.io/afowler/sentiment-neuron
model = Model(open('model/charList.txt').read(), 1, mustRestore=True, dump=False)
#
@app.route('/')
def index():
    return 'Worked!'
    
@app.route('/api/infer_model')
def infer_model():
    img = preprocess(cv2.imread('data/test.png', cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
#   
    return jsonify({'recognized': str(recognized[0]), 'probability': str(probability[0])})

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    content = request.json
    image = np.reshape(np.frombuffer(base64.decodebytes(content['image'].encode('utf-8')), dtype=np.uint8), (-1, 1))
#    
    img = preprocess(cv2.imdecode(image, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
#
    return jsonify({'recognized': str(recognized[0]), 'probability': str(probability[0])})
#
if __name__ == '__main__':
    app.run(debug=True)
