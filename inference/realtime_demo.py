import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
import collections
import json

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import STGCN
from features.facial_features import FACE_INDICES

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("..", "models", "stgcn_wlasl100_final.pth")
JSON_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
NUM_CLASSES = 100
WINDOW_SIZE = 45        # We analyze the last 45 frames (1.5 seconds)
MODEL_INPUT_SIZE = 64   # The model expects 64 frames (we pad the rest)
CONFIDENCE_THRESHOLD = 0.2  # Lowered to 20% because the model is naturally cautious

# --- LOAD LABELS ---
try:
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)[:NUM_CLASSES] # Same limit as training
        labels = [entry['gloss'] for entry in data]
    print(f"Loaded {len(labels)} classes.")
except FileNotFoundError:
    print(f"ERROR: Could not find JSON at {JSON_PATH}")
    labels = []

# --- HELPER: EXTRACTION WITH ROBUST NORMALIZATION ---
def extract_landmarks(results):
    # 1. Pose (33)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))
    
    # 2. Hands & Face (Standard)
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
        
    # --- ROBUST NORMALIZATION LOGIC ---
    # Strategy: Center the skeleton around the Nose (Index 0).
    # If Nose is missing, fallback to the midpoint of Shoulders (Indices 11 & 12).
    
    center = pose[0] # Default center is Nose
    
    # Check if Nose is all zeros (missing)
    if not np.any(center):
        # Check if Shoulders exist
        if np.any(pose[11]) and np.any(pose[12]):
            center = (pose[11] + pose[12]) / 2.0
            
    # Apply subtraction if we found a valid center
    keypoints = np.concatenate([pose, lh, rh, face])
    if np.any(center):
        keypoints = keypoints - center
        
    return keypoints

# --- HELPER: EYEBROW LOGIC ---
def get_expression_tag(face_landmarks):
    if not face_landmarks: return ""
    
    try:
        # Access raw MediaPipe landmarks
        left_eye_top = face_landmarks.landmark[159].y
        left_eye_bottom = face_landmarks.landmark[145].y
        left_eyebrow = face_landmarks.landmark[105].y
        
        eye_open = abs(left_eye_top - left_eye_bottom)
        brow_dist = abs(left_eyebrow - left_eye_top)
        
        if eye_open < 0.001: return "" # Blink
        
        ratio = brow_dist / eye_open
        
        if ratio > 0.9: return "[QUESTION]" # Eyebrows raised
        if ratio < 0.3: return "[INTENSE]"  # Eyebrows furrowed
    except:
        pass
    return ""

# --- MAIN INFERENCE LOOP ---
def run_inference():
    # 1. Load Model
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

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    
    # MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    # Buffer
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    current_prediction = "Waiting..."
    current_confidence = 0.0
    expression_text = ""

    print("--- STARTING REALTIME DEMO ---")
    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        
        # Draw Logic
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 3. Extract & Buffer
        # (Normalization is now handled inside this function)
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        
        # 4. Check Expression
        new_expr = get_expression_tag(results.face_landmarks)
        if new_expr: expression_text = new_expr

        # 5. Predict
        if len(sequence) == WINDOW_SIZE:
            # Prepare Input
            input_data = np.array(list(sequence)) # (45, 109, 3)
            
            # Pad to 64 frames
            pad_len = MODEL_INPUT_SIZE - WINDOW_SIZE
            if pad_len > 0:
                padding = np.zeros((pad_len, 109, 3))
                input_data = np.concatenate([input_data, padding], axis=0)
            
            # Reshape for ST-GCN: (Batch, Channel, Time, Node, Person)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            input_tensor = input_tensor.permute(2, 0, 1)          # (3, 64, 109)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1) # (1, 3, 64, 109, 1)
            
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, idx = torch.max(probs, 1)
                
                # Check Threshold
                if conf.item() > CONFIDENCE_THRESHOLD: 
                    current_prediction = labels[idx.item()]
                    current_confidence = conf.item()
                else:
                    # Optional: Keep showing the last word if it was recent, 
                    # or show "..." if nothing is detected.
                    current_prediction = "..."

        # 6. UI Display
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        
        text = f"{expression_text} {current_prediction} ({int(current_confidence*100)}%)"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign2Sound Euphoria', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()