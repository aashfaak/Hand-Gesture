import cv2
import mediapipe as mp
import numpy as np
import pyautogui  # For controlling PowerPoint

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Track hand position for swipe gestures
prev_x = None

def recognize_gesture(hand_landmarks):
    global prev_x
    landmarks = hand_landmarks.landmark
    
    thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, 
                          landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
    index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
                          landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    middle_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, 
                           landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, 
                      landmarks[mp_hands.HandLandmark.WRIST].y])
    
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
    
    if thumb_middle_dist < 0.05:
        return "Zoom Out"
    elif thumb_index_dist < 0.05:
        return "Zoom In"
    
    if prev_x is not None:
        if wrist[0] - prev_x > 0.05:  # Adjusted sensitivity for faster response
            prev_x = wrist[0]
            return "Slide Next"
        elif prev_x - wrist[0] > 0.05:
            prev_x = wrist[0]
            return "Slide Previous"
    prev_x = wrist[0]
    
    if thumb_tip[1] > wrist[1] and index_tip[1] < wrist[1] and middle_tip[1] < wrist[1]:
        return "Slide Close"
    
    return "Unknown Gesture"

cap = cv2.VideoCapture(0)
last_gesture = None

time_threshold = 0.5  # Time delay to avoid repeated actions
import time
last_action_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if gesture != last_gesture and time.time() - last_action_time > time_threshold:
                if gesture == "Slide Next":
                    pyautogui.press('right')
                elif gesture == "Slide Previous":
                    pyautogui.press('left')
                elif gesture == "Slide Close":
                    pyautogui.hotkey('alt', 'f4')
                elif gesture == "Zoom In":
                    pyautogui.hotkey('ctrl', '+')
                elif gesture == "Zoom Out":
                    pyautogui.hotkey('ctrl', '-')
                last_gesture = gesture
                last_action_time = time.time()

    cv2.imshow("Real-Time Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
