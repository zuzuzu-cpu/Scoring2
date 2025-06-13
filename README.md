# Judo Score Predictor

This application uses TensorFlow and MediaPipe to analyze live judo matches and predict scores based on pose detection.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Real-time video feed processing
- Pose detection using MediaPipe
- Score prediction based on detected poses
- Live updates of predictions
- Clean and responsive web interface

## Note

The current version includes a placeholder for the actual prediction logic. To implement real judo technique recognition and scoring, you'll need to:

1. Train a TensorFlow model on judo technique data
2. Replace the `predict_score()` function in `app.py` with your trained model
3. Add more sophisticated pose analysis for specific judo techniques

## Requirements

- Python 3.8+
- Webcam or video input device
- Modern web browser 