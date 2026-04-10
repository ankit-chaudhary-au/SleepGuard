import cv2
from config import CAMERA_INDEX


class Camera:
    # it class manages camera for sleepguard
    # it opens camera when it is initialized
    
    
    def __init__(self):
        # it opens webcam using the index defined in config.py.
        # it displays runtime error if webcam cannot be open
        
        
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {CAMERA_INDEX}. "
                f"Try changing CAMERA_INDEX in config.py to 1 or 2."
            )
        
        # it reads one frame to check if webcam is working
        ret, _ = self.cap.read()
        if not ret:
            raise RuntimeError(
                "Camera opened but could not read a frame. "
                "Check that no other application is using the webcam."
            )

        print(f"[Camera] Webcam opened successfully on index {CAMERA_INDEX}.")
        
        
    def get_frame(self):
        # it reads and returs frame from webcam
      
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    
    def get_frame_size(self):
        # it returns width and height of the webcam frame
      
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    
    def release(self):
       # it closes the webcam frame when system is shut down
        self.cap.release()
        cv2.destroyAllWindows()
        print("[Camera] Webcam released.")

