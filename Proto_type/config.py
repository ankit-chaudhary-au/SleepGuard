# mediapipe model
MODEL_PATH = "face_landmarker.task"

# camera settings
CAMERA_INDEX = 0 # 0 is default camera, set it to 1 or 2 if wrong camera opens

# EAR ( eye aspect ratio) settings
EAR_THRESHOLD = 0.25 # if EAR drops below this than eye is closed

# seconds before alarm is triggred
CLOSED_SECONDS = 3.0

# MAR (mouth aspect ratio)
MAR_THRESHOLD = 0.60 # if MAR rises over this than it is a yawn


# number of yawn before it shows drowsiness warning
YAWN_COUNT_THRESHOLD = 3

#head position settings
HEAD_PITCH_THRESHOLD = 20.0

# alert settings
ALARM_SOUND = "alarm.mp3"

# color settings
COLOUR_ALERT   = (0, 255, 0)    # Green  driver is alert
COLOUR_WARNING = (0, 165, 255)  # Orange yawn warning
COLOUR_DROWSY  = (0, 0, 255)    # Red  drowsy alarm triggered
COLOUR_INFO    = (0, 255, 255)  # Yellow information text

# Font Settings
FONT_SCALE_LARGE = 1.0
FONT_SCALE_SMALL = 0.65
FONT_THICKNESS   = 2

