from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import mediapipe as mp
import threading
import queue
import json
import os
import time

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the TFLite model and classes
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('classes.json', 'r') as f:
    classes = json.load(f)

# Global variables
frame_queue = queue.Queue(maxsize=2)
prediction_queue = queue.Queue(maxsize=2)
current_prediction = {"score": "None", "technique": "None"}
camera_source = 0  # Default to first camera
camera_thread = None
camera_active = False

def get_available_cameras():
    """Get list of available camera sources"""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def process_frame(frame):
    """Process a single frame and return pose landmarks"""
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get pose landmarks
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Convert landmarks to numpy array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks
    return None

def predict_score(landmarks):
    """Predict score based on pose landmarks using the loaded model"""
    if landmarks is not None:
        # Reshape landmarks for model input
        landmarks = landmarks.reshape(1, -1).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], landmarks)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        
        # Get the class name from the prediction
        class_name = [k for k, v in classes.items() if v == predicted_class][0]
        
        return {
            "score": class_name,
            "technique": class_name
        }
    return {"score": "None", "technique": "None"}

def video_processing_thread():
    """Background thread for video processing"""
    global camera_active
    try:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            # If camera is not available, use a test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_image, "Camera not available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            while camera_active:
                if not frame_queue.full():
                    frame_queue.put(test_image)
                if not prediction_queue.full():
                    prediction_queue.put({"score": "None", "technique": "None"})
                time.sleep(0.1)
            return

        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            landmarks = process_frame(frame)
            prediction = predict_score(landmarks)
            
            # Update queues
            if not frame_queue.full():
                frame_queue.put(frame)
            if not prediction_queue.full():
                prediction_queue.put(prediction)
                
        cap.release()
    except Exception as e:
        print(f"Error in video processing thread: {e}")
        # If any error occurs, keep sending a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, f"Camera Error: {str(e)}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while camera_active:
            if not frame_queue.full():
                frame_queue.put(test_image)
            if not prediction_queue.full():
                prediction_queue.put({"score": "None", "technique": "None"})
            time.sleep(0.1)

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    available_cameras = get_available_cameras()
    return render_template('index.html', cameras=available_cameras)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    if not prediction_queue.empty():
        current_prediction = prediction_queue.get()
    return jsonify(current_prediction)

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global camera_source, camera_thread, camera_active
    
    # Stop current camera thread if running
    if camera_active:
        camera_active = False
        if camera_thread:
            camera_thread.join()
    
    # Get new camera source from request
    new_source = request.json.get('camera_source', 0)
    try:
        new_source = int(new_source)
        camera_source = new_source
        camera_active = True
        
        # Start new camera thread
        camera_thread = threading.Thread(target=video_processing_thread)
        camera_thread.daemon = True
        camera_thread.start()
        
        return jsonify({"status": "success", "message": f"Camera source set to {new_source}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Start video processing thread
    camera_active = True
    camera_thread = threading.Thread(target=video_processing_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Use PORT env variable for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 