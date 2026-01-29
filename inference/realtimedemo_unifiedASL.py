import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
import collections
import json
import time

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import STGCN
from features.facial_features import FACE_INDICES

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("..", "models", "stgcn_unified_final.pth")
CLASSES_PATH = os.path.join("..","data", "classes.json")
CONFIDENCE_THRESHOLD = 0.5

# Triggers
REST_THRESHOLD_Y = 0.75  # Hands below 75% of screen height = Rest
CALIBRATION_FRAMES = 40  # Time to measure your "Neutral" face

# --- NMM: RELATIVE GEOMETRY (Distance Invariant) ---
def get_eyebrow_ratio(face_landmarks):
    """
    Calculates Ratio = (Eyebrow-Eye Distance) / (Eye-Nose Distance).
    This works even if you move closer/further from the camera.
    """
    if not face_landmarks: return 0.0
    
    try:
        # MediaPipe Indices
        # Right Eyebrow Middle: 334
        # Right Eye Top: 386
        # Nose Tip: 1
        
        eyebrow = np.array([face_landmarks.landmark[334].x, face_landmarks.landmark[334].y])
        eye = np.array([face_landmarks.landmark[386].x, face_landmarks.landmark[386].y])
        nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y])
        
        brow_eye_dist = np.linalg.norm(eyebrow - eye)
        eye_nose_dist = np.linalg.norm(eye - nose)
        
        if eye_nose_dist < 0.001: return 0.0
        
        return brow_eye_dist / eye_nose_dist
    except:
        return 0.0

# --- MAIN SYSTEM ---
class SignLanguageSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
        # Load Labels
        try:
            with open(CLASSES_PATH, 'r') as f:
                self.labels = json.load(f)
            print(f"✅ Loaded {len(self.labels)} Classes")
        except:
            print(f"❌ ERROR: {CLASSES_PATH} not found. Run export_labels.py!")
            sys.exit()

        # State Variables
        self.sequence = collections.deque(maxlen=45)
        self.gloss_buffer = []
        self.last_word = None
        self.cooldown = 0
        self.rest_state = True
        
        # NMM State
        self.calibrating = True
        self.calibration_data = []
        self.neutral_ratio = 0.0
        self.is_question = False

    def load_model(self):
        # Update num_class if your training data size changed!
        self.model = STGCN(num_class=124, in_channels=3, edge_strategy='spatial')
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model Loaded")
        else:
            print("❌ Model file missing!")
            sys.exit()

    def check_rest_pose(self, results):
        """Returns True if hands are dropped below threshold or not visible"""
        left_down = True
        right_down = True
        
        if results.left_hand_landmarks:
            if results.left_hand_landmarks.landmark[0].y < REST_THRESHOLD_Y: 
                left_down = False 
            
        if results.right_hand_landmarks:
            if results.right_hand_landmarks.landmark[0].y < REST_THRESHOLD_Y: 
                right_down = False 
            
        # If no hands visible, assume rest
        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            return True
            
        return left_down and right_down

    def extract_landmarks(self, results):
        # Standard 109-point extraction
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        else: pose = np.zeros((33, 3))
        
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        else: lh = np.zeros((21, 3))
        
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        else: rh = np.zeros((21, 3))
        
        if results.face_landmarks:
            face_all = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
            face = face_all[FACE_INDICES] 
        else: face = np.zeros((len(FACE_INDICES), 3))
            
        center = pose[0]
        keypoints = np.concatenate([pose, lh, rh, face])
        if np.any(center): keypoints = keypoints - center
        return keypoints

    def run(self):
        cap = cv2.VideoCapture(0)
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        print("--- SYSTEM READY ---")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # --- PHASE 1: CALIBRATION ---
            if self.calibrating:
                cv2.putText(image, "CALIBRATING... LOOK NEUTRAL", (50, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if results.face_landmarks:
                    ratio = get_eyebrow_ratio(results.face_landmarks)
                    self.calibration_data.append(ratio)
                    
                    if len(self.calibration_data) > CALIBRATION_FRAMES:
                        self.neutral_ratio = np.mean(self.calibration_data)
                        self.calibrating = False
                        print(f"✅ Calibrated. Neutral Ratio: {self.neutral_ratio:.3f}")
                
                cv2.imshow('Sign2Sound', image)
                if cv2.waitKey(10) == ord('q'): break
                continue

            # --- PHASE 2: PROCESSING ---
            
            # A. Detect Question Expression
            if results.face_landmarks:
                curr_ratio = get_eyebrow_ratio(results.face_landmarks)
                # If eyebrows raised 15% higher than normal -> QUESTION
                if curr_ratio > (self.neutral_ratio * 1.15):
                    self.is_question = True
                    cv2.putText(image, "[?]", (550, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # B. Check Rest Pose
            current_rest = self.check_rest_pose(results)
            
            if not current_rest:
                # ACTIVE SIGNING
                self.rest_state = False
                keypoints = self.extract_landmarks(results)
                self.sequence.append(keypoints)
                
                if self.cooldown > 0: self.cooldown -= 1
                
                if len(self.sequence) == 45 and self.cooldown == 0:
                    # Prepare Tensor
                    inp = np.array(list(self.sequence))
                    pad = np.zeros((64 - 45, 109, 3))
                    inp = np.concatenate([inp, pad], axis=0)
                    inp_tensor = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(self.device)
                    
                    with torch.no_grad():
                        out = self.model(inp_tensor)
                        prob = torch.softmax(out, dim=1)
                        conf, idx = torch.max(prob, 1)
                        
                        if conf.item() > CONFIDENCE_THRESHOLD:
                            word = self.labels[idx.item()]
                            # Simple dedup
                            if word != self.last_word:
                                self.gloss_buffer.append(word)
                                self.last_word = word
                                self.cooldown = 20 # Wait 20 frames
                                print(f"Detected: {word}")

            else:
                # REST DETECTED
                if not self.rest_state:
                    # Just finished a sentence
                    if self.gloss_buffer:
                        # --- FINAL OUTPUT FORMAT ---
                        final_output = self.gloss_buffer.copy()
                        if self.is_question:
                            final_output.insert(0, "[QUESTION]")
                        
                        # PRINT TO CONSOLE FOR TEAMMATE'S SLM
                        print(f"\n>>> OUTPUT: {final_output}")
                        
                        # Reset for next sentence
                        self.gloss_buffer = []
                        self.last_word = None
                        self.is_question = False
                        
                self.rest_state = True
                self.sequence.clear()

            # UI
            display_text = " ".join(self.gloss_buffer)
            cv2.rectangle(image, (0,0), (640, 45), (30, 30, 30), -1)
            cv2.putText(image, f"Buffer: {display_text}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status_color = (0, 0, 255) if current_rest else (0, 255, 0)
            cv2.circle(image, (620, 30), 8, status_color, -1)

            cv2.imshow('Sign2Sound', image)
            if cv2.waitKey(10) == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys = SignLanguageSystem()
    sys.run()