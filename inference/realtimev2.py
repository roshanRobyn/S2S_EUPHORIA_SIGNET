import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
import collections
import json
from collections import Counter

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import STGCN
from features.facial_features import FACE_INDICES

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("..", "models", "stgcn_wlasl100_final.pth")
JSON_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
NUM_CLASSES = 100
WINDOW_SIZE = 45        # History for the Model (1.5 seconds)
MODEL_INPUT_SIZE = 64   # Input size for the Model
CONFIDENCE_THRESHOLD = 0.2 

# --- NEW: SMOOTHING CONFIGURATION ---
PREDICTION_BUFFER_SIZE = 10  # We wait for 10 consistent frames
VOTE_THRESHOLD = 0.7         # 70% of the buffer must agree (7 out of 10)

# --- LOAD LABELS ---
try:
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)[:NUM_CLASSES]
        labels = [entry['gloss'] for entry in data]
    print(f"Loaded {len(labels)} classes.")
except FileNotFoundError:
    print(f"ERROR: Could not find JSON at {JSON_PATH}")
    labels = []

# --- NEW CLASS: PREDICTION SMOOTHER ---
class PredictionSmoother:
    def __init__(self, buffer_size=10, vote_threshold=0.7):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.vote_threshold = vote_threshold
        self.last_stable_prediction = "..."

    def add_and_get_stable(self, raw_prediction):
        # 1. Add new guess to history
        self.buffer.append(raw_prediction)
        
        # 2. Count votes
        counter = Counter(self.buffer)
        most_common, count = counter.most_common(1)[0]
        
        # 3. Check if winner has enough votes
        total = len(self.buffer)
        ratio = count / total
        
        if ratio >= self.vote_threshold and most_common != "...":
            self.last_stable_prediction = most_common
            return most_common
        
        # 4. If chaotic, keep showing the old one (or "..." if you prefer)
        return self.last_stable_prediction

# --- HELPER: EXTRACTION ---
def extract_landmarks(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))
    
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    else:
        lh = np.zeros((21, 3))

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    else:
        rh = np.zeros((21, 3))

    if results.face_landmarks:
        face_all = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
        face = face_all[FACE_INDICES] 
    else:
        face = np.zeros((len(FACE_INDICES), 3))
        
    # Robust Normalization
    center = pose[0]
    if not np.any(center) and np.any(pose[11]) and np.any(pose[12]):
        center = (pose[11] + pose[12]) / 2.0
            
    keypoints = np.concatenate([pose, lh, rh, face])
    if np.any(center):
        keypoints = keypoints - center
        
    return keypoints

def get_expression_tag(face_landmarks):
    if not face_landmarks: return ""
    try:
        left_eye_top = face_landmarks.landmark[159].y
        left_eye_bottom = face_landmarks.landmark[145].y
        left_eyebrow = face_landmarks.landmark[105].y
        
        eye_open = abs(left_eye_top - left_eye_bottom)
        brow_dist = abs(left_eyebrow - left_eye_top)
        
        if eye_open < 0.001: return ""
        ratio = brow_dist / eye_open
        
        if ratio > 0.9: return "[QUESTION]"
        if ratio < 0.3: return "[INTENSE]"
    except:
        pass
    return ""

# --- MAIN INFERENCE LOOP ---
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCN(num_class=NUM_CLASSES, in_channels=3, edge_strategy='spatial')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model Loaded Successfully!")
    else:
        print(f"ERROR: Model weights not found at {MODEL_PATH}")
        return

    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    
    # Initialize Smoother
    smoother = PredictionSmoother(buffer_size=PREDICTION_BUFFER_SIZE, vote_threshold=VOTE_THRESHOLD)
    
    final_display_text = "Waiting..."
    current_confidence = 0.0
    expression_text = ""

    print("--- STARTING SMOOTH DEMO ---")
    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        
        new_expr = get_expression_tag(results.face_landmarks)
        if new_expr: expression_text = new_expr

        if len(sequence) == WINDOW_SIZE:
            input_data = np.array(list(sequence))
            pad_len = MODEL_INPUT_SIZE - WINDOW_SIZE
            if pad_len > 0:
                padding = np.zeros((pad_len, 109, 3))
                input_data = np.concatenate([input_data, padding], axis=0)
            
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
            input_tensor = input_tensor.to(device)
            
            # 1. Get Raw Prediction
            raw_pred = "..."
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, idx = torch.max(probs, 1)
                
                if conf.item() > CONFIDENCE_THRESHOLD: 
                    raw_pred = labels[idx.item()]
                    current_confidence = conf.item()
            
            # 2. Smooth it!
            final_display_text = smoother.add_and_get_stable(raw_pred)

        # UI Display
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        
        # Display the STABLE prediction, not the raw one
        text = f"{expression_text} {final_display_text} ({int(current_confidence*100)}%)"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign2Sound Euphoria', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()