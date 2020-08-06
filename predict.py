'''
pip install flask
pip install Pillow
pip install keras
pip install tensorflow
'''

import io
import base64
import numpy as np
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
	global model
	model = load_model('BeeCNN.h5')

def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")

	image =  image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	return image

get_model()

@app.route("/predict", methods=['POST'])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, target_size=(224,224))
	prediction = model.predict(processed_image)
	
	response = {
		'Prediction': {
			'Sana': str(prediction[0][0]),
			'Reina': str(prediction[0][1]),
			'Varroa': str(prediction[0][2])
		}
	}

	return jsonify(response)