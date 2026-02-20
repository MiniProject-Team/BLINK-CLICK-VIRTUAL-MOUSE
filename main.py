"""
Blink-Click Virtual Mouse – Final Version
---------------------------------------
Features:
✔ Cursor moves using eye iris movement
✔ Left click → Long blink
✔ Right click → Double long blink
✔ Colorful camera view
✔ Lighting enhancement
✔ Noise reduction
✔ Increased cursor sensitivity
✔ Smooth cursor control
✔ Eye-fatigue rest reminder

Press ESC to exit
"""

import cv2
import time
import math
import mediapipe as mp
import pyautogui


# -----------------------------
# Utility Functions
# -----------------------------
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_point(landmarks, index, w, h):
    return (
        int(landmarks[index].x * w),
        int(landmarks[index].y * h)
    )


def eye_aspect_ratio(landmarks, w, h, top, bottom, left, right):
    t = get_point(landmarks, top, w, h)
    b = get_point(landmarks, bottom, w, h)
    l = get_point(landmarks, left, w, h)
    r = get_point(landmarks, right, w, h)

    vertical = distance(t, b)
    horizontal = distance(l, r)

    if horizontal == 0:
        return 1

    return vertical / horizontal


# -----------------------------
# Main Program
# -----------------------------
def main():

    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Iris landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    # Eye blink landmarks
    L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133
    R_TOP, R_BOTTOM, R_LEFT, R_RIGHT = 386, 374, 362, 263

    # Cursor smoothing & sensitivity
    smooth_x, smooth_y = screen_w / 2, screen_h / 2
    smooth_factor = 0.35     # Faster response
    cursor_speed = 2.5       # Increased sensitivity

    # Blink detection
    blink_threshold = 0.18
    intentional_blink_time = 0.4

    blink_start_time = 0
    blink_detected = False

    # Double blink
    last_blink_time = 0
    double_blink_gap = 0.6

    # Rest reminder
    session_start = time.time()
    rest_interval = 60  # seconds

    print("Blink-Click Virtual Mouse started.")
    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # -----------------------------
        # Lighting Enhancement (Colorful)
        # -----------------------------
        alpha = 1.4   # Contrast
        beta = 20     # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Noise reduction
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # -----------------------------
        # Face Mesh Processing
        # -----------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            # -----------------------------
            # Cursor Movement
            # -----------------------------
            def iris_center(indices):
                pts = [get_point(landmarks, i, w, h) for i in indices]
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                return cx, cy

            lx, ly = iris_center(LEFT_IRIS)
            rx, ry = iris_center(RIGHT_IRIS)

            gaze_x = (lx + rx) / 2
            gaze_y = (ly + ry) / 2

            nx = gaze_x / w
            ny = gaze_y / h

            target_x = nx * screen_w * cursor_speed
            target_y = ny * screen_h * cursor_speed

            smooth_x = (1 - smooth_factor) * smooth_x + smooth_factor * target_x
            smooth_y = (1 - smooth_factor) * smooth_y + smooth_factor * target_y

            pyautogui.moveTo(smooth_x, smooth_y)

            # Draw iris centers
            cv2.circle(frame, (int(lx), int(ly)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(rx), int(ry)), 3, (0, 255, 0), -1)

            # -----------------------------
            # Blink Detection
            # -----------------------------
            left_ear = eye_aspect_ratio(
                landmarks, w, h,
                L_TOP, L_BOTTOM, L_LEFT, L_RIGHT
            )

            right_ear = eye_aspect_ratio(
                landmarks, w, h,
                R_TOP, R_BOTTOM, R_LEFT, R_RIGHT
            )

            avg_ear = (left_ear + right_ear) / 2
            now = time.time()

            if avg_ear < blink_threshold:

                if not blink_detected:
                    blink_start_time = now
                    blink_detected = True

            else:
                if blink_detected:
                    blink_duration = now - blink_start_time

                    if blink_duration > intentional_blink_time:

                        # Double blink → Right click
                        if (now - last_blink_time) < double_blink_gap:
                            pyautogui.rightClick()
                        else:
                            pyautogui.click()

                        last_blink_time = now

                    blink_detected = False

            # Show blink ratio
            cv2.putText(frame,
                        f"Blink Ratio: {avg_ear:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

        # -----------------------------
        # Rest Reminder
        # -----------------------------
        if time.time() - session_start > rest_interval:
            cv2.putText(frame,
                        "Take Eye Rest!",
                        (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        # Display window
        cv2.imshow("Blink Click Virtual Mouse", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()