# YOLO Inference Server

This project provides a FastAPI server for performing inference with YOLO models for sign detection and lane segmentation.

## Features

- Real-time sign detection using YOLO models
- Lane segmentation with overlay visualization
- Support for both image and video inference
- Web interface for easy testing
- API endpoints for integration with other systems

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── server.py                  # FastAPI: load model, infer images, return annotated images
├── app/
│   ├── __init__.py
│   ├── config.py              # read .env, model paths, general configuration
│   ├── models.py              # initialize & cache 4 models: sign/lane × best/last
│   ├── infer_sign.py          # inference logic + draw bbox for signs (detect)
│   ├── infer_lane.py          # inference logic + draw mask/overlay for lanes (segment)
│   └── utils.py               # utilities: read images, convert bytes, color conversion…
├── static/
│   ├── index.html             # upload interface
│   ├── script.js              # call API /infer/sign, /infer/lane
│   └── style.css
```

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python server.py
   ```

The server will start on `http://127.0.0.1:8000/static/index.html` by default.

## Web Interface

After starting the server, you can access the web interface at:

```
http://127.0.0.1:8000/static/index.html
```

The web interface allows you to:
- Upload images for sign detection
- Upload images for lane segmentation
- Upload videos for sign detection
- Upload videos for lane segmentation
- Select between 'best' and 'last' model variants
- Adjust confidence thresholds

## Requirements

- Python 3.7+
- ultralytics==8.3.40
- fastapi==0.115.5
- uvicorn[standard]==0.32.0
- python-multipart==0.0.9
- Pillow==10.4.0
- opencv-python-headless==4.10.0.84
- onnxruntime>=1.17.0

## Model Structure

The models should be organized in the following structure:

```
exports_yolo/
├── sign/
│   ├── best/
│   │   └── weights/
│   │       ├── best.onnx
│   │       └── ...
│   └── last/
│       └── weights/
│           ├── last.onnx
│           └── ...
└── lane/
    ├── best/
    │   └── weights/
    │       ├── best.onnx
    │       └── ...
    └── last/
        └── weights/
            ├── last.onnx
            └── ...
```

