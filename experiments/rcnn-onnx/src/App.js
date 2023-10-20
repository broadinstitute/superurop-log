import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css"

const App = () => {
    const [session, setSession] = useState(null);
    const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
    const [image, setImage] = useState(null);
    const inputImage = useRef(null);
    const imageRef = useRef(null);
    const canvasRef = useRef(null);

    // configs
    const modelName = "FasterRCNN-12-qdq.onnx"
    const modelInputShape = [1, 3, ]
}











import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
