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
VIDEO_PATH = "whowantspizza.mp4" 
MODEL_PATH = os.path.join("..", "models", "stgcn_unified_final.pth")
CLASSES_PATH = os.path.join("..", "data", "classes.json")

# --- 🎛️ THE MAGIC TOGGLES (Change these if confidence is low) ---
MIRROR_X = False      # Try TRUE first. (Flips Left/Right)
REMOVE_Z = True      # Try TRUE first. (Flattens 3D to 2D to remove noise)
INVERT_Y = False     # Keep FALSE usually. (Flips Up/Down)
SCALE_FACTOR = 4.5   # Keep at 4.5 (Since we confirmed Math Check passed)

class VideoInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
        try:
            with open(CLASSES_PATH, 'r') as f:
                self.labels = json.load(f)
            print(f"✅ Classes Loaded: {len(self.labels)}")
        except:
            sys.exit()

        self.sequence = collections.deque(maxlen=45)

    def load_model(self):
        self.model = STGCN(num_class=124, in_channels=3, edge_strategy='spatial')
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

    def extract_landmarks(self, results):
        # 1. Extract
        if results.pose_landmarks: pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        else: pose = np.zeros((33, 3))
        if results.left_hand_landmarks: lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        else: lh = np.zeros((21, 3))
        if results.right_hand_landmarks: rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        else: rh = np.zeros((21, 3))
        if results.face_landmarks:
            face_all = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
            face = face_all[FACE_INDICES] 
        else: face = np.zeros((len(FACE_INDICES), 3))
            
        # 2. APPLY TOGGLES (Pre-Normalization)
        keypoints = np.concatenate([pose, lh, rh, face])
        
        if MIRROR_X:
            keypoints[:, 0] = -keypoints[:, 0] # Flip X axis
            
        if INVERT_Y:
            keypoints[:, 1] = -keypoints[:, 1] # Flip Y axis

        # 3. CENTER (Re-calculate center after flipping)
        # We assume center is the Nose (Index 0)
        center = keypoints[0] 
        # If nose is 0,0 (missing), try average
        if not np.any(center): center = np.mean(keypoints, axis=0)
        
        keypoints = keypoints - center

        # 4. SCALE
        # Recalculate shoulder dist from the keypoints array directly
        # Indices in concatenated array: Pose is 0-32. Left Shoulder=11, Right=12
        shoulder_dist = np.linalg.norm(keypoints[11] - keypoints[12])
        
        scale = 1.0
        if shoulder_dist > 0.01:
            scale = shoulder_dist * SCALE_FACTOR

        keypoints = keypoints / scale 
        
        # 5. REMOVE Z (Post-Normalization)
        if REMOVE_Z:
            keypoints[:, 2] = 0 # Flatten to 2D
            
        return keypoints

    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)
        
        print(f"--- RUNNING WITH: Mirror={MIRROR_X}, NoZ={REMOVE_Z} ---")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            self.sequence.append(self.extract_landmarks(results))
            
            # Predict every 5 frames
            if len(self.sequence) == 45 and frame_count % 5 == 0:
                inp = np.array(list(self.sequence))
                pad = np.zeros((64 - 45, 109, 3))
                inp = np.concatenate([inp, pad], axis=0)
                inp_tensor = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(self.device)
                
                with torch.no_grad():
                    out = self.model(inp_tensor)
                    prob = torch.softmax(out, dim=1)
                    values, indices = torch.topk(prob, 3)
                    
                    # Print Top 1
                    top1 = self.labels[indices[0][0]]
                    conf1 = values[0][0].item()
                    
                    # Log significant detections
                    if conf1 > 0.1:
                         print(f"Frame {frame_count}: {top1} ({conf1:.2f})")

        cap.release()
        print("--- DONE ---")

if __name__ == "__main__":
    app = VideoInference()
    app.run()