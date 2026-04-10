import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from config import (
    MODEL_PATH,
    EAR_THRESHOLD,
    CLOSED_SECONDS,
    MAR_THRESHOLD,
    YAWN_COUNT_THRESHOLD,
    HEAD_PITCH_THRESHOLD,
)


 # eye landmarks
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# mouth landmarks
MOUTH = [13, 14, 78, 308, 82, 312]

# nose tip and chin used for head tilt 
NOSE_TIP = 1
CHIN     = 152
FOREHEAD = 10

class DetectionEngine:
    # this class detects the drowsiness by processing each frame and returning if person is 
    # drowsiness by eye closure, yawning, and head position.
    
    
    def __init__(self):
        # it initilized mediapipe face landmark in video mode
        
        options = FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        print(" MediaPipe Face Landmarker initialised.")
        
    
         # tracking state
        self.eye_closed_start  = None    # it shows when eye start to close
        self.alarm_triggered   = False   # it stops alarm to triger every frame
        self.yawn_count        = 0       # it counts number of yawn 
        self.currently_yawning = False   # it tracks yawn state if the person is yawning or not
        self.frame_timestamp   = 0      
        
    # EAR calculation (EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||))
    def _calculate_ear(self, eye_points):
        
       # it calculates eye expect ratio of one eye
       
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1  = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical_2  = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal  = np.linalg.norm(np.array(p1) - np.array(p4))
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    
    def _get_avg_ear(self, landmarks, w, h):
        # it calculated average ear of both eyes
        
        def extract(indices):
            return [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in indices
            ]

        left_ear  = self._calculate_ear(extract(LEFT_EYE))
        right_ear = self._calculate_ear(extract(RIGHT_EYE))
        return (left_ear + right_ear) / 2.0
    
    
    # MAR calculation
    def _get_mar(self, landmarks, w, h):
        # it calculated Mouth aspect ration to detect yawning
        # high MAR means person is yawning
        
        def pt(idx):
            return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

        top    = pt(MOUTH[0])
        bottom = pt(MOUTH[1])
        left   = pt(MOUTH[2])
        right  = pt(MOUTH[3])

        vertical   = np.linalg.norm(np.array(top) - np.array(bottom))
        horizontal = np.linalg.norm(np.array(left) - np.array(right))

        if horizontal == 0:
            return 0.0
        return vertical / horizontal
    
    
    
    # head position estimation
    def _get_head_pitch(self, landmarks, h):
        # it estimates head position forward or backward tilt using facial landmarks
        
        nose_y     = landmarks[NOSE_TIP].y * h
        chin_y     = landmarks[CHIN].y * h
        forehead_y = landmarks[FOREHEAD].y * h

        face_height = chin_y - forehead_y
        if face_height == 0:
            return 0.0

        # ration how far nose from top of the face, when face tilt forward ratio increases
        ratio = (nose_y - forehead_y) / face_height

        pitch = (ratio - 0.45) * 100
        return pitch
    
    
    
    # main process
    def process_frame(self, frame):
          # it process single webcam frame through full detection pipeline
          
        h, w, _ = frame.shape
          
        # converting BGR frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
          
        # increases timestamp manually
        self.frame_timestamp += 33 # 30 fps = 33ms per frame
          
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp)
          
          
        # default return values when face is not detected 
        output = {
            "face_detected"  : False,
            "avg_ear"        : 0.0,
            "mar"            : 0.0,
            "head_pitch"     : 0.0,
            "drowsy"         : False,
            "yawn_detected"  : False,
            "yawn_count"     : self.yawn_count,
            "head_nodding"   : False,
            "elapsed_closed" : 0.0,
            "alarm_triggered": self.alarm_triggered,
            "landmarks"      : None,
            "frame_w"        : w,
            "frame_h"        : h,
        }
        
          
          
        if not result.face_landmarks:
                # it there is no face in frame it resets eye timer
                self.eye_closed_start = None
                self.alarm_triggered  = False
                return output
            
        
        landmarks = result.face_landmarks[0]
        output["face_detected"] = True
        output["landmarks"]     = landmarks

        # EAR eye closed detection
        avg_ear = self._get_avg_ear(landmarks, w, h)
        output["avg_ear"] = round(avg_ear, 3)
            
        elapsed_closed = 0.0
        if avg_ear < EAR_THRESHOLD:
            if self.eye_closed_start is None:
                self.eye_closed_start = time.time()
            elapsed_closed = time.time() - self.eye_closed_start

            if elapsed_closed >= CLOSED_SECONDS:
                output["drowsy"]          = True
                output["alarm_triggered"] = True
                self.alarm_triggered      = True
        else:
                self.eye_closed_start = None
                self.alarm_triggered  = False

        output["elapsed_closed"] = round(elapsed_closed, 1)
            
            
            
        # MAR yawn detection
        mar = self._get_mar(landmarks, w, h)
        output["mar"] = round(mar, 3)

        if mar > MAR_THRESHOLD:
            output["yawn_detected"] = True
            if not self.currently_yawning:
                    self.yawn_count        += 1
                    self.currently_yawning  = True
        else:
            self.currently_yawning = False

        output["yawn_count"] = self.yawn_count
            
            
        # head position detection
        pitch = self._get_head_pitch(landmarks, h)
        output["head_pitch"]   = round(pitch, 1)
        output["head_nodding"] = pitch > HEAD_PITCH_THRESHOLD

        return output









        

