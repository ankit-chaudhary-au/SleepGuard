import cv2
import pygame
from config import(
    ALARM_SOUND,
    COLOUR_ALERT,
    COLOUR_WARNING,
    COLOUR_DROWSY,
    COLOUR_INFO,
    FONT_SCALE_LARGE,
    FONT_SCALE_SMALL,
    FONT_THICKNESS,
)
 # eye landmarks
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# mouth landmarks
MOUTH_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


class AlertSystem:
    # this class plays alert noise when ever eye is close
    
    def __init__(self):
        # this initializes pygame mixer to play audio
        # if audio file is not found it prints warning but camera alert works 
        
        pygame.mixer.init()
        self.alarm_playing = False
        
        try:
            pygame.mixer.music.load(ALARM_SOUND)
            print(f"[Alert] Alarm sound loaded: {ALARM_SOUND}")
        except Exception as e:
            print(f"[Alert] WARNING - Could not load alarm sound: {e}")
            print("[Alert] Visual alerts will still work. Check alarm.mp3 is in the same folder.")
            
            
    def trigger_alarm(self):
        # it plays alarm sound if it is not playing,it stops alarm to restart every frame
        
        if not self.alarm_playing:
            try:
                pygame.mixer.music.play(-1)  
                self.alarm_playing = True
                print("[Alert] ALARM TRIGGERED - drowsiness detected.")
            except Exception as e:
                print(f"[Alert] Could not play alarm: {e}")
                
                
    def stop_alarm(self):
        # it stops the alarm when eyes are open again
        
        if self.alarm_playing:
            pygame.mixer.music.stop()
            self.alarm_playing = False


    def draw_landmarks(self, frame, detection_result):
        # it draws landmarks on the face
        # small green dots for all 478 mesh points, 6 eye landmards on each eye,
        # yellow circle highlights mouth landmarks
        
        
        landmarks = detection_result.get("landmarks")
        if landmarks is None:
            return frame
        
        w = detection_result["frame_w"]
        h = detection_result["frame_h"]
        
        #face mesh points
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # left eye landmarks
        for idx in LEFT_EYE:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

        # right eye landmarks
        for idx in RIGHT_EYE:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)
            
        # eye outline by connecting 6 points
        def draw_eye_outline(indices, colour):
            pts = [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in indices
            ]
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], colour, 1)

        draw_eye_outline(LEFT_EYE,  (0, 255, 255))
        draw_eye_outline(RIGHT_EYE, (0, 255, 255))
        
        # mouth outlines
        mouth_pts = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in MOUTH_OUTLINE
        ]
        for i in range(len(mouth_pts)):
            cv2.line(
                frame,
                mouth_pts[i],
                mouth_pts[(i + 1) % len(mouth_pts)],
                (0, 165, 255), 1
            )

        return frame
    
    
    def draw_overlay(self, frame, detection_result):
        # it shows live status overlay on webcam frame
        # it shows all detection metrics and current system status 
        
        h, w, _ = frame.shape
        
        # it displays infromation palen in left corner
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (340, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if not detection_result["face_detected"]:
            # it shows No face detected warning
            cv2.putText(frame, "NO FACE DETECTED", (20, 50),
                        font, FONT_SCALE_SMALL, COLOUR_WARNING, FONT_THICKNESS)
            return frame
        
        # it draws face landmarks and eye and mouth outline
        self.draw_landmarks(frame, detection_result)

        avg_ear    = detection_result["avg_ear"]
        mar        = detection_result["mar"]
        pitch      = detection_result["head_pitch"]
        yawn_count = detection_result["yawn_count"]
        elapsed    = detection_result["elapsed_closed"]
        drowsy     = detection_result["drowsy"]
        yawning    = detection_result["yawn_detected"]
        nodding    = detection_result["head_nodding"]
        
        
        cv2.putText(frame, f"EAR:   {avg_ear:.3f}", (20, 40),
                    font, FONT_SCALE_SMALL, COLOUR_INFO, FONT_THICKNESS)

        cv2.putText(frame, f"MAR:   {mar:.3f}", (20, 70),
                    font, FONT_SCALE_SMALL, COLOUR_INFO, FONT_THICKNESS)

        cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (20, 100),
                    font, FONT_SCALE_SMALL, COLOUR_INFO, FONT_THICKNESS)

        cv2.putText(frame, f"Yawns: {yawn_count}", (20, 130),
                    font, FONT_SCALE_SMALL, COLOUR_INFO, FONT_THICKNESS)

        if elapsed > 0:
            cv2.putText(frame, f"Eyes closed: {elapsed:.1f}s", (20, 160),
                        font, FONT_SCALE_SMALL, COLOUR_WARNING, FONT_THICKNESS)
            
        # it shows status on top right corner
        if drowsy:
            status_text   = "DROWSY - WAKE UP!"
            status_colour = COLOUR_DROWSY
            self.trigger_alarm()
            
            
               # it flashes red border line when drowsy
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOUR_DROWSY, 8)

        elif yawning or nodding or yawn_count >= 3:
            status_text   = "WARNING - FATIGUED"
            status_colour = COLOUR_WARNING
            self.stop_alarm()
        else:
            status_text   = "ALERT"
            status_colour = COLOUR_ALERT
            self.stop_alarm()
            
            
        # it shows status 
        text_size = cv2.getTextSize(
            status_text, font, FONT_SCALE_LARGE, FONT_THICKNESS
        )[0]
        text_x = (w - text_size[0]) // 2

        cv2.putText(frame, status_text, (text_x, h - 20),
                    font, FONT_SCALE_LARGE, status_colour, FONT_THICKNESS + 1)
        
        
        badge_y = 210
        if yawning:
            cv2.putText(frame, "YAWN DETECTED", (20, badge_y),
                        font, FONT_SCALE_SMALL, COLOUR_WARNING, FONT_THICKNESS)
            badge_y += 30

        if nodding:
            cv2.putText(frame, "HEAD NODDING", (20, badge_y),
                        font, FONT_SCALE_SMALL, COLOUR_WARNING, FONT_THICKNESS)

        return frame
    
    def quit(self):
        # it quits pygame mixer when exited
        pygame.mixer.quit()
        print("[Alert] Alert system shut down.")








        
        