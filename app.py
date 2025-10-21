from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import requests
import cv2
import numpy as np
import base64
import io
import os
import uuid
import threading
import time
from collections import deque
import math

app = Flask(__name__)
CORS(app)

# Increase max upload size to 500MB and optimize for large files
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_EXTENSIONS'] = ['.mp4', '.avi', '.mov', '.mkv']
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Model handling (lazy load). If model file is not present, the server will
# continue running and return 503 for processing endpoints until the model is
# available. You can set an environment variable MODEL_URL to a direct
# download link (e.g., an S3 presigned URL or a Google Drive direct link
# served via a small proxy). The service will try to download the model on
# first load attempt.
model = None
model_path = os.path.join(os.path.dirname(__file__), "best_02.pt")
if not os.path.exists(model_path):
    # fallback for local dev
    model_path = "../../best_02.pt"


def download_model_if_requested(dest_path: str) -> bool:
    """If MODEL_URL is set in the environment, try to download it to dest_path.
    Returns True on success, False otherwise."""
    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        return False

    try:
        app.logger.info(f"Downloading model from MODEL_URL: {model_url}")
        with requests.get(model_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        app.logger.info("Model downloaded successfully")
        return True
    except Exception as e:
        app.logger.exception("Failed to download model: %s", e)
        return False


def load_model() -> bool:
    """Attempt to load the YOLO model. Returns True if model is loaded."""
    global model
    if model is not None:
        return True

    # If file doesn't exist, try to download if configured
    if not os.path.exists(model_path):
        app.logger.warning(f"Model not found at {model_path}")
        # try to download using MODEL_URL
        try:
            dest_dir = os.path.dirname(model_path)
            os.makedirs(dest_dir, exist_ok=True)
        except Exception:
            pass
        downloaded = download_model_if_requested(model_path)
        if not downloaded and not os.path.exists(model_path):
            app.logger.warning("Model not available and MODEL_URL not provided or download failed")
            return False

    try:
        app.logger.info(f"Loading model from {model_path} ...")
        model = YOLO(model_path)
        app.logger.info("Model loaded successfully")
        return True
    except Exception as e:
        app.logger.exception("Exception while loading model: %s", e)
        model = None
        return False

# Try to load model at startup but do not crash on failure
try:
    _ = load_model()
except Exception:
    app.logger.exception("Initial model load attempt failed")

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Tracking configuration
small_threshold = 5000
large_threshold = 15000
max_disappeared = 30

# Active processing sessions
processing_sessions = {}

class PotholeDetector:
    def __init__(self):
        self.next_pothole_id = 0
        self.pothole_trackers = {}
        self.pothole_count = {"Small": 0, "Medium": 0, "Large": 0}
        self.final_pothole_count = {"Small": 0, "Medium": 0, "Large": 0}
        self.detected_areas = []
        self.risk_levels = {"Low": 0, "Medium": 0, "High": 0}
        
    def calculate_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w // 2, y + h // 2)
    
    def calculate_distance(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def get_risk_level(self, size_category, position_y, road_height):
        road_center = road_height / 2
        center_factor = 1 - (abs(position_y - road_center) / road_center)
        
        if size_category == "Large":
            risk_base = 2
        elif size_category == "Medium":
            risk_base = 1
        else:
            risk_base = 0
        
        risk_adjusted = risk_base + center_factor
        
        if risk_adjusted >= 2:
            return "High", (0, 0, 255)
        elif risk_adjusted >= 1:
            return "Medium", (0, 165, 255)
        else:
            return "Low", (0, 255, 0)
    
    def update_trackers(self, new_potholes):
        for pothole_id in list(self.pothole_trackers.keys()):
            self.pothole_trackers[pothole_id]["disappeared"] += 1
            if self.pothole_trackers[pothole_id]["disappeared"] > max_disappeared:
                if pothole_id not in [p for p in self.final_pothole_count]:
                    self.final_pothole_count[self.pothole_trackers[pothole_id]["final_size"]] += 1
                del self.pothole_trackers[pothole_id]
        
        if len(new_potholes) == 0:
            return
        
        if len(self.pothole_trackers) == 0:
            for i in range(len(new_potholes)):
                self.pothole_trackers[self.next_pothole_id] = {
                    "centroid": new_potholes[i]["centroid"],
                    "disappeared": 0,
                    "size": new_potholes[i]["size"],
                    "contour": new_potholes[i]["contour"],
                    "area": new_potholes[i]["area"],
                    "risk": new_potholes[i]["risk"],
                    "max_area": new_potholes[i]["area"],
                    "final_size": new_potholes[i]["size"],
                    "final_risk": new_potholes[i]["risk"]
                }
                self.next_pothole_id += 1
        else:
            tracker_ids = list(self.pothole_trackers.keys())
            tracker_centroids = [self.pothole_trackers[tid]["centroid"] for tid in tracker_ids]
            distance_matrix = []
            for new_pothole in new_potholes:
                distances = []
                for tracker_centroid in tracker_centroids:
                    d = self.calculate_distance(new_pothole["centroid"], tracker_centroid)
                    distances.append(d)
                distance_matrix.append(distances)
            
            used_trackers = set()
            used_potholes = set()
            
            while True:
                min_distance = float("inf")
                min_pothole_idx = -1
                min_tracker_idx = -1
                
                for i in range(len(new_potholes)):
                    if i in used_potholes:
                        continue
                    for j in range(len(tracker_ids)):
                        if j in used_trackers:
                            continue
                        if distance_matrix[i][j] < min_distance:
                            min_distance = distance_matrix[i][j]
                            min_pothole_idx = i
                            min_tracker_idx = j
                
                if min_pothole_idx == -1 or min_distance > 50:
                    break
                    
                tracker_id = tracker_ids[min_tracker_idx]
                self.pothole_trackers[tracker_id]["centroid"] = new_potholes[min_pothole_idx]["centroid"]
                self.pothole_trackers[tracker_id]["disappeared"] = 0
                self.pothole_trackers[tracker_id]["size"] = new_potholes[min_pothole_idx]["size"]
                self.pothole_trackers[tracker_id]["contour"] = new_potholes[min_pothole_idx]["contour"]
                self.pothole_trackers[tracker_id]["area"] = new_potholes[min_pothole_idx]["area"]
                self.pothole_trackers[tracker_id]["risk"] = new_potholes[min_pothole_idx]["risk"]
                
                if new_potholes[min_pothole_idx]["area"] > self.pothole_trackers[tracker_id]["max_area"]:
                    self.pothole_trackers[tracker_id]["max_area"] = new_potholes[min_pothole_idx]["area"]
                    self.pothole_trackers[tracker_id]["final_size"] = new_potholes[min_pothole_idx]["size"]
                    self.pothole_trackers[tracker_id]["final_risk"] = new_potholes[min_pothole_idx]["risk"]
                
                used_trackers.add(min_tracker_idx)
                used_potholes.add(min_pothole_idx)
            
            for i in range(len(new_potholes)):
                if i not in used_potholes:
                    self.pothole_trackers[self.next_pothole_id] = {
                        "centroid": new_potholes[i]["centroid"],
                        "disappeared": 0,
                        "size": new_potholes[i]["size"],
                        "contour": new_potholes[i]["contour"],
                        "area": new_potholes[i]["area"],
                        "risk": new_potholes[i]["risk"],
                        "max_area": new_potholes[i]["area"],
                        "final_size": new_potholes[i]["size"],
                        "final_risk": new_potholes[i]["risk"]
                    }
                    self.next_pothole_id += 1
    
    def process_frame(self, frame):
        h, w, _ = frame.shape
        self.pothole_count = {"Small": 0, "Medium": 0, "Large": 0}
        self.risk_levels = {"Low": 0, "Medium": 0, "High": 0}
        
        results = model.predict(frame, verbose=False)
        overlay = frame.copy()
        new_potholes = []
        
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if masks is not None:
                masks = masks.data.cpu()
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        centroid = self.calculate_centroid(contour)
                        
                        if area < small_threshold:
                            size_category = "Small"
                        elif area > large_threshold:
                            size_category = "Large"
                        else:
                            size_category = "Medium"
                        
                        risk_level, risk_color = self.get_risk_level(size_category, centroid[1], h)
                        self.pothole_count[size_category] += 1
                        self.detected_areas.append(area)
                        self.risk_levels[risk_level] += 1
                        
                        new_potholes.append({
                            "centroid": centroid,
                            "size": size_category,
                            "contour": contour,
                            "area": area,
                            "risk": risk_level
                        })
        
        self.update_trackers(new_potholes)
        
        # Draw potholes
        for pothole_id, data in self.pothole_trackers.items():
            if data["disappeared"] == 0:
                contour = data["contour"]
                final_size = data["final_size"]
                centroid = data["centroid"]
                final_risk = data["final_risk"]
                
                # Only mark potholes in bottom 25% of screen
                if centroid[1] >= h * 0.75:
                    if final_risk == "High":
                        color = (0, 0, 255)
                    elif final_risk == "Medium":
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    
                    cv2.drawContours(overlay, [contour], -1, color, -1)
                    cv2.polylines(frame, [contour], True, color=color, thickness=3)
                
                x, y, width, height = cv2.boundingRect(contour)
                label = f"ID:{pothole_id} {final_size} ({final_risk})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, centroid, 4, (255, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Add statistics overlay
        stats_y = 30
        cv2.rectangle(frame, (10, 10), (300, 140), (30, 30, 30), -1)
        cv2.putText(frame, "Pothole Statistics", (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        stats_y += 30
        cv2.putText(frame, f"Small: {self.pothole_count['Small']} (Total: {self.final_pothole_count['Small']})", 
                   (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        stats_y += 25
        cv2.putText(frame, f"Medium: {self.pothole_count['Medium']} (Total: {self.final_pothole_count['Medium']})", 
                   (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        stats_y += 25
        cv2.putText(frame, f"Large: {self.pothole_count['Large']} (Total: {self.final_pothole_count['Large']})", 
                   (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame

def process_video(video_path, output_path, session_id):
    try:
        detector = PotholeDetector()
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                processed_frame = detector.process_frame(frame)
                out.write(processed_frame)
            else:
                out.write(frame)
            
            frame_count += 1
            
            # Update progress
            if session_id in processing_sessions:
                processing_sessions[session_id]['progress'] = int((frame_count / total_frames) * 100)
        
        cap.release()
        out.release()
        
        # Calculate final totals including tracked potholes that disappeared
        for pothole_id in list(detector.pothole_trackers.keys()):
            if pothole_id not in [p for p in detector.final_pothole_count]:
                detector.final_pothole_count[detector.pothole_trackers[pothole_id]["final_size"]] += 1
        
        total = detector.final_pothole_count['Small'] + detector.final_pothole_count['Medium'] + detector.final_pothole_count['Large']
        
        if session_id in processing_sessions:
            processing_sessions[session_id]['status'] = 'completed'
            processing_sessions[session_id]['output_path'] = output_path
            processing_sessions[session_id]['total_detections'] = total
            processing_sessions[session_id]['small_potholes'] = detector.final_pothole_count['Small']
            processing_sessions[session_id]['medium_potholes'] = detector.final_pothole_count['Medium']
            processing_sessions[session_id]['large_potholes'] = detector.final_pothole_count['Large']
            
    except Exception as e:
        if session_id in processing_sessions:
            processing_sessions[session_id]['status'] = 'failed'
            processing_sessions[session_id]['error'] = str(e)

@app.route('/health', methods=['GET'])
def health():
    # Report whether the model is loaded so callers can understand degraded state
    model_available = load_model()
    status = "healthy" if model_available else "degraded"
    return jsonify({
        "status": status,
        "model_loaded": bool(model_available),
        "message": "Pothole Detection API is running"
    })

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    # Ensure model is available before accepting processing jobs
    if not load_model():
        return jsonify({"error": "Model not loaded. Set MODEL_URL or upload the model to the server."}), 503

    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Initialize session BEFORE saving (so upload progress can be tracked)
    processing_sessions[session_id] = {
        'status': 'uploading',
        'progress': 0,
        'output_path': None,
        'error': None,
        'total_detections': 0,
        'small_potholes': 0,
        'medium_potholes': 0,
        'large_potholes': 0
    }
    
    # Save uploaded video with progress tracking
    video_filename = f"{session_id}_{video_file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    try:
        # Stream save for large files
        video_file.save(video_path)
        processing_sessions[session_id]['status'] = 'processing'
    except Exception as e:
        processing_sessions[session_id]['status'] = 'failed'
        processing_sessions[session_id]['error'] = f"Upload failed: {str(e)}"
        return jsonify({"error": "Upload failed", "details": str(e)}), 500
    
    # Output path
    output_filename = f"{session_id}_processed.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Start processing in background
    thread = threading.Thread(target=process_video, args=(video_path, output_path, session_id))
    thread.daemon = True  # Make thread daemon so it doesn't block shutdown
    thread.start()
    
    return jsonify({
        "session_id": session_id,
        "message": "Video uploaded, processing started"
    })

@app.route('/process_status/<session_id>', methods=['GET'])
def process_status(session_id):
    if session_id not in processing_sessions:
        return jsonify({"error": "Invalid session ID"}), 404
    
    session = processing_sessions[session_id]
    return jsonify({
        "status": session['status'],
        "progress": session['progress'],
        "error": session.get('error'),
        "total_detections": session.get('total_detections', 0),
        "small_potholes": session.get('small_potholes', 0),
        "medium_potholes": session.get('medium_potholes', 0),
        "large_potholes": session.get('large_potholes', 0)
    })

@app.route('/download_video/<session_id>', methods=['GET'])
def download_video(session_id):
    if session_id not in processing_sessions:
        return jsonify({"error": "Invalid session ID"}), 404
    
    session = processing_sessions[session_id]
    
    if session['status'] != 'completed':
        return jsonify({"error": "Video processing not completed"}), 400
    
    output_path = session['output_path']
    
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 404
    
    return send_file(output_path, as_attachment=True, download_name=f"processed_{session_id}.mp4")

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Process a single frame for real-time camera detection"""
    try:
        # Ensure model is loaded before processing a frame
        if not load_model():
            return jsonify({"error": "Model not loaded. Set MODEL_URL or upload the model to the server."}), 503

        if 'frame' not in request.files:
            return jsonify({"error": "No frame provided"}), 400
        
        frame_file = request.files['frame']
        
        # Read image
        img_bytes = frame_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Create detector for single frame
        detector = PotholeDetector()
        processed_frame = detector.process_frame(frame)
        
        # Encode back to image
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "processed_frame": frame_base64,
            "statistics": {
                "current": detector.pothole_count,
                "total": detector.final_pothole_count
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
