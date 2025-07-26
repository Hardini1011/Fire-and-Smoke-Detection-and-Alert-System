# Fire & Smoke Detection and Alert System
This project is a real-time **Fire and Smoke Detection System** using **YOLOv8** for object detection and **Deep SORT** for object tracking. It features a user-friendly **Streamlit interface** for video upload and visual result display, with automatic **Twilio SMS alerts** when fire or smoke is detected in the video.

## Project Overview
- Detects **fire** and **smoke** from uploaded video files
- Tracks detected objects frame-to-frame using **Deep SORT**
- Sends **SMS alerts** instantly using **Twilio**
- Visualizes bounding boxes, confidence scores, and object IDs
- Outputs processed video and JSON log of detections

## Tech Stack
- YOLOv8 (Ultralytics) – for object detection
- Deep SORT – for object tracking
- OpenCV – for video frame processing
- Streamlit – for the web interface
- Twilio API – for SMS alert system
- Python – as the main programming language

## Requirements
Install the following dependencies:
```bash
pip install streamlit opencv-python-headless ultralytics numpy stqdm deep_sort_realtime python-dotenv twilio

## Contributors
Soha Chauhan
Hardini Dalwadi
