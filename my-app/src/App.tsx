import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import logo from "./assets/icons/xmark-solid.svg";

function App() {

  // Parse result into respective data type
  interface Result {
    age: number;
    gender: string;
  }

  // Manage all the application states
  const [result, setResult] = useState<Result>({ age: 18, gender: "Male" });
  const [image, setImage] = useState<File | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Age and Gender Prediction</h1>
        <div className="image-input-container">{imageDisplay()}</div>
        <button
          className="predict-button"
          onClick={() => {
            // Predict age and gender of a given image
            if (image != null && !isPredicting) {
              setIsPredicting(true);

              var data = new FormData();

              data.append("img", image);
              
              // Send a POST API to the predict endpoint
              axios
                .post("http://127.0.0.1:5000/predict", data, {
                  headers: {
                    "Content-Type": "multipart/form-data",
                  },
                })
                .then((response) => {
                  setResult(response.data);
                  setIsPredicting(false);
                });
            }
          }}
        >
          {isPredicting ? "Predicting" : "Predict"}
        </button>
        <div className="result-container">
          <div className="result">
            <p className="result-text">Predicted Age :</p>
            <p className="result-text">Predicted Result:</p>
          </div>

          <div className="result">
            <p className="result-text">{result.age}</p>
            <p className="result-text">{result.gender}</p>
          </div>
        </div>
      </header>
    </div>
  );

  function imageDisplay() {
    if (image == null) {
      return (
        <div>
          <input
            name="img"
            className="select-button"
            type="file"
            accept="image/png, image/jpeg"
            onChange={(event) => {
              // Receive the image file input
              const files = event.currentTarget.files;

              if (files?.length != null && files.length !== 0) {
                setImage(files[0]);
              }
            }}
          ></input>
        </div>
      );
    } else {
      return (
        <div>
          <img
            className="reset_button"
            src={logo}
            alt="close"
            onClick={() => setImage(null)}
          />
          <img
            className="preview"
            src={URL.createObjectURL(image)}
            alt="preview"
          />
        </div>
      );
    }
  }
}

export default App;
