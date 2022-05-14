from io import BytesIO
from PIL import Image
import requests
from flask import Flask, request
from flask_cors import CORS

from src.model import AgeGenderPredictor


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

age_group = {
    0: "Between 0 - 3",
    1: "Between 4 - 7",
    2: "Between 8 - 14",
    3: "Between 15 - 23",
    4: "Between 24 - 35",
    5: "Between 36 - 45",
    6: "Between 46 - 55",
    7: "Above 60",
}


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        model = AgeGenderPredictor(vgg_path="./pretrained/vgg16.weight")
        model.load_model("./checkpoints/epoch_5")
        model.eval()

        img_file = request.files['img']

        img = Image.open(BytesIO(img_file.stream.read()))

        age, gender = model.predict(img)

        print(gender.item())

    return {
        'age': age_group[age.item()],
        'gender': "Female" if gender.item() == 0 else "Male",
    }
