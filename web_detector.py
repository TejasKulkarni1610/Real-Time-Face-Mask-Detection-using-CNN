import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from flask import Flask, Response, render_template

# --- CONFIGURATION ---
# IMPORTANT: Model path based on your successful InceptionV3 run
MODEL_PATH = "local_mask_detector_InceptionV3_SGD.h5" 

# Use os.path.join for reliable path construction and os.path.abspath 
# to force the system to look in the exact location relative to the script's directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths using cross-platform join method
PROTOTXT_PATH = os.path.join(SCRIPT_DIR, "face_detector", "deploy.prototxt")
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

CONFIDENCE_THRESHOLD = 0.5
IMG_DIMS = (224, 224)

# --- FLASK SETUP ---
app = Flask(__name__)

# --- INITIALIZATION ---
# Load the face detector model and the mask detector model
print(f"[INFO] Looking for prototxt at: {PROTOTXT_PATH}")
print(f"[INFO] Looking for weights at: {WEIGHTS_PATH}")

# Check if files actually exist before trying to load them
if not os.path.exists(PROTOTXT_PATH):
    raise FileNotFoundError(f"Face detector prototxt not found at: {PROTOTXT_PATH}")
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Face detector weights not found at: {WEIGHTS_PATH}")

print("[INFO] Loading face detector model...")
net = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)

# Load your custom-trained mask model
print("[INFO] Loading mask detector model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}. Check your .h5 file name.")

# Use custom_objects dictionary to load models trained using Keras 3/TensorFlow 2.x conventions if needed.
model = load_model(MODEL_PATH)
# Warm-up the model by making a quick prediction on dummy data
model.predict(np.zeros((1, *IMG_DIMS, 3), dtype="float32")) 


# Initialize the video stream
print("[INFO] Starting video stream...")
# src=0 refers to the default camera
vs = VideoStream(src=0).start()
time.sleep(2.0) 


def detect_and_predict_mask(frame, net, model):
    # Grab the frame dimensions and construct a blob
    (h, w) = frame.shape[:2]
    # Resize to 300x300 for the DNN model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and get detections
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            
            # Check if the ROI is valid
            if face.size > 0:
                # Pre-process the face for the mask detector model
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, IMG_DIMS)
                
                # Reshape and preprocess for model
                face = np.expand_dims(face, axis=0) # Add batch dimension
                face = preprocess_input(face)      # MobileNetV2 preprocessing function

                # Add the face and bounding box to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Only make predictions if at least one face was detected
    if len(faces) > 0:
        # Concatenate faces into a batch and predict
        faces_batch = np.vstack(faces)
        preds = model.predict(faces_batch, verbose=0)
        
        # Convert predictions from batch prediction format back to list of predictions
        preds = [pred for pred in preds] 

    return (locs, preds)

def generate_frames():
    # Loop over frames from the video stream
    while True:
        frame = vs.read()
        if frame is None:
            continue
            
        frame = imutils.resize(frame, width=800)

        # Detect faces and predict mask status
        (locs, preds) = detect_and_predict_mask(frame, net, model)

        # Loop over the detected face locations and their corresponding predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            
            # Determine the class label and color
            # NOTE: We assume the mask generator orders classes alphabetically: 0: 'with_mask', 1: 'without_mask'
            mask_prob, nomask_prob = pred
            
            if mask_prob > nomask_prob:
                label = "Mask OK"
                color = (0, 255, 0) # Green (BGR format)
                prob = mask_prob
            else:
                label = "Please Wear Mask"
                color = (0, 0, 255) # Red
                prob = nomask_prob

            # Display the label and bounding box rectangle
            label_text = f"{label}: {prob * 100:.2f}%"

            cv2.putText(frame, label_text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        # Yield the output frame in the byte format for MJPEG streaming
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    # Return the rendered template (templates/index.html)
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # Return the response generated by the generator function
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    
    # Upon exiting, release the video stream
    print("[INFO] Cleaning up...")
    vs.stop()