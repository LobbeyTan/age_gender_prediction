from io import BytesIO
from PIL import Image
from flask import Flask, request
from flask_cors import CORS

from src.model import AgeGenderPredictor

# Create a Flask app
app = Flask(__name__)

# Set CORS to be able to access from all origins
CORS(app, resources={r"/predict": {"origins": "*"}})

# Age group mapping
age_group = {
    0: "Between 0 - 3",
    1: "Between 4 - 7",
    2: "Between 8 - 14",
    3: "Between 15 - 23",
    4: "Between 24 - 35",
    5: "Between 36 - 45",
    6: "Between 46 - 55",
    7: "Above 56",
}

# Configure predict endpoint with POST method only


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load the predictor
        model = AgeGenderPredictor(vgg_path="./pretrained/vgg16.weight")
        # Load the trained weights
        model.load_model("./pretrained/model")
        # Set the model to evaluation mode
        model.eval()

        # Get the image file
        img_file = request.files['img']

        # Using PIL to open image read from stream of bytes
        img = Image.open(BytesIO(img_file.stream.read()))

        # Get the prediction from the model
        age, gender = model.predict(img)

    # Return the result in the form of map
    return {
        'age': age_group[age.item()],
        'gender': "Female" if gender.item() == 0 else "Male",
    }
