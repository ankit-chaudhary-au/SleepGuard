import cv2
import sys
from camera    import Camera
from detection import DetectionEngine
from alert     import AlertSystem

def main():
    print("=" * 55)
    print("  SleepGuard - Real-Time Driver Fatigue Detection")
    print("  ICT304 Assignment 1")
    print("=" * 55)
    print("\nStarting system... press Q or ESC to quit.\n")
    
    
    # initiallizing all modules
    try:
        camera    = Camera()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("Check your webcam is connected and not in use by another app.")
        sys.exit(1)

    try:
        detector  = DetectionEngine()
    except Exception as e:
        print(f"\n[ERROR] Could not initialise detection engine: {e}")
        print("Make sure face_landmarker.task is in the same folder as main.py.")
        sys.exit(1)

    alert = AlertSystem()

    print("\n[SleepGuard] System running. Monitoring person...\n")
    
    # main processing loop
    while True:
        # getting frame from webcam
        frame = camera.get_frame()
        if frame is None:
            print("[WARNING] Could not read frame. Retrying...")
            continue

        # running detection on frame
        result = detector.process_frame(frame)
        
        
        # drawing overlay and handlng alerts
        frame = alert.draw_overlay(frame, result)

        # displaying frame
        cv2.imshow("SleepGuard - Driver Fatigue Detection", frame)

        # printing status in terminal every 30 frames
        if detector.frame_timestamp % (33 * 30) == 0:
            status = "DROWSY" if result["drowsy"] else "ALERT"
            print(
                f"[Status] {status} | "
                f"EAR: {result['avg_ear']:.3f} | "
                f"MAR: {result['mar']:.3f} | "
                f"Pitch: {result['head_pitch']:.1f}° | "
                f"Yawns: {result['yawn_count']}"
            )

        # checking for quiting key esc or Q
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\n[SleepGuard] Quit signal received. Shutting down...")
            break

    # shutting down all modules
    camera.release()
    alert.quit()
    print("[SleepGuard] System stopped. Goodbye.")


if __name__ == "__main__":
    main()


