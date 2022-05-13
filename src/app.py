from io import BytesIO
from PIL import Image
import requests
from flask import Flask, request
from flask_cors import CORS

from src.model import AgeGenderPredictor


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        model = AgeGenderPredictor()
        model.load_model("./checkpoints/epoch_30")
        model.eval()

        img_file = request.files['img']

        img = Image.open(BytesIO(img_file.stream.read()))

        age, gender = model.predict(img)

    return {
        'age': age.item(),
        'gender': "Female" if gender.item() == 0 else "Male",
    }
