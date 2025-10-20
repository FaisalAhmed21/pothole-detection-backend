# Pothole Detection Backend API

Flask-based REST API for processing videos and real-time frames for pothole detection.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure `best_02.pt` model file is in the parent directory

3. Run the server:
```bash
python app.py
```

The API will run on `http://localhost:5000`

## API Endpoints

### 1. Health Check
- **GET** `/health`
- Returns server status

### 2. Process Video
- **POST** `/process_video`
- Upload a video file for processing
- Returns a session ID

### 3. Check Processing Status
- **GET** `/process_status/<session_id>`
- Check the progress of video processing

### 4. Download Processed Video
- **GET** `/download_video/<session_id>`
- Download the processed video

### 5. Process Single Frame (Real-time)
- **POST** `/process_frame`
- Upload a single frame for real-time detection
- Returns processed frame with detections

## Testing with curl

```bash
# Health check
curl http://localhost:5000/health

# Upload video
curl -X POST -F "video=@path/to/video.mp4" http://localhost:5000/process_video

# Check status
curl http://localhost:5000/process_status/<session_id>

# Download processed video
curl http://localhost:5000/download_video/<session_id> -o processed.mp4
```

## Deploy for Mobile Access

### Option 1: Local Network
Run the server and access it from your phone on the same WiFi network:
```bash
python app.py
```
Access via: `http://<your-pc-ip>:5000`

### Option 2: ngrok (for testing)
```bash
# Install ngrok
# Then run:
ngrok http 5000
```
Use the provided URL in your mobile app

### Option 3: Cloud Deployment
Deploy to:
- Heroku
- AWS EC2
- Google Cloud Run
- DigitalOcean

## Notes
- Videos are temporarily stored in `uploads/` folder
- Processed videos are stored in `outputs/` folder
- Clean up old files periodically to save space
