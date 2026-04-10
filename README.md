# SleepGuard — Real-Time Driver Fatigue Detection System

**ICT304 Assignment 1 — Murdoch University**  
**Team Members:**
- Ankit Chaudhary Tharu (34493239) — Member 1
- Sanchita Arora (35080343) — Member 2  
- Ananya Singh (34774675) — Member 3

---

## Project Description

SleepGuard is a real-time driver fatigue detection system that uses a standard webcam to continuously monitor a driver's alertness. The system detects drowsiness through three simultaneous methods:

- Eye Aspect Ratio (EAR) — detects prolonged eye closure
- Mouth Aspect Ratio (MAR) — detects yawning
- Head Pose Estimation — detects forward head nodding

When drowsiness is detected, the system triggers an immediate audio alarm. In the final project, it will also automatically contact emergency services with the vehicle's GPS location in the event of a crash.


## System Requirements

- Python 3.9, 3.10, or 3.11 (Python 3.12+ not recommended)
- Webcam (built-in or USB)
- Operating System: Windows 10/11, macOS 12+

## Installation and Setup

### Step 1 — Install dependencies
pip install -r requirements.txt

### Step 2 — Download the MediaPipe model file
Download `face_landmarker.task` from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Place it in the `proto_type/` folder alongside `main.py`

### Step 3 — Add alarm sound
Place `alarm.mp3` in the `proto_type/` folder alongside `main.py`

### Step 4 — Run the system
cd proto_type
python main.py

Press **Q** or **ESC** to quit.


---

## How It Works

1. `main.py` starts the system and runs the main processing loop
2. `camera.py` opens the webcam and captures frames
3. `detection.py` processes each frame through MediaPipe Face Mesh and computes EAR, MAR, and head pitch
4. `alert.py` draws the live overlay and plays the alarm when drowsiness is detected
5. `config.py` stores all configurable thresholds — change values here without touching any other file

## References

- Soukupova, T., & Cech, J. (2016). Real-time eye blink detection using facial landmarks. *21st Computer Vision Winter Workshop*
- Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*
- NASA. (2017). *NASA Systems Engineering Handbook* (SP-2016-6105 Rev2)

