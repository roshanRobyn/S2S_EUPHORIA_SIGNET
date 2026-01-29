import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
import json

# --- CONFIGURATION ---
VIDEO_QUEUE = [
    "who.mp4", 
    "eat.mp4", 
    "pizza.mp4"
]

MODEL_PATH = os.path.join("..", "models", "stgcn_wlasl100_final.pth") 
CLASSES_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
MODEL_INPUT_SIZE = 64  # Match training size

# --- SYSTEM SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import STGCN
from features.facial_features import FACE_INDICES

class PipelineDemo:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing Standard Pipeline on {self.device}...")
        
        self.load_classes()
        self.load_model()
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def load_classes(self):
        try:
            with open(CLASSES_PATH, 'r') as f:
                data = json.load(f)
                # Matches your working script's loading logic exactly
                # It loads the first 100 classes from the JSON
                if isinstance(data, list) and isinstance(data[0], dict):
                     self.labels = [entry['gloss'] for entry in data]
                else:
                     self.labels = data
                
                # Ensure we strictly match the model's 100 classes
                self.labels = self.labels[:100] 
                print(f"✅ Classes Loaded: {len(self.labels)}")
        except Exception as e:
            print(f"❌ Error loading {CLASSES_PATH}: {e}")
            sys.exit()

    def load_model(self):
        self.model = STGCN(num_class=100, in_channels=3, edge_strategy='spatial')
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model Loaded")
        else:
            print(f"❌ Model not found: {MODEL_PATH}")
            sys.exit()

    def extract_keypoints(self, results):
        # EXACT COPY OF YOUR WORKING LOGIC
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
            
        keypoints = np.concatenate([pose, lh, rh, face])
        
        # NORMALIZATION: Subtract Nose Only (Matches working script)
        nose = keypoints[0]
        if np.any(nose):
            keypoints = keypoints - nose
            
        # NO SCALING, NO MIRRORING, NO Z-REMOVAL
        return keypoints

    def process_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"⚠️ Warning: File {video_path} not found. Skipping.")
            return None

        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            kps = self.extract_keypoints(results)
            frames_data.append(kps)
        
        cap.release()

        if len(frames_data) == 0: return None

        # --- PRE-PROCESSING (Matches working script EXACTLY) ---
        data_numpy = np.array(frames_data) # (T, 109, 3)
        T, V, C = data_numpy.shape
        
        # Center Crop / Pad Logic
        final_input = np.zeros((MODEL_INPUT_SIZE, V, C))
        if T > MODEL_INPUT_SIZE:
            start = (T - MODEL_INPUT_SIZE) // 2
            final_input = data_numpy[start:start+MODEL_INPUT_SIZE, :, :]
        else:
            final_input[:T, :, :] = data_numpy

        # Transpose to (1, C, T, V, 1)
        final_input = np.transpose(final_input, (2, 0, 1))
        final_input = np.expand_dims(final_input, axis=-1)
        
        inp_tensor = torch.from_numpy(final_input).float().unsqueeze(0)
        inp_tensor = inp_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            out = self.model(inp_tensor)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, 1)
            
            word = self.labels[idx.item()]
            confidence = conf.item()
            
            return word, confidence

    def run(self):
        final_sentence = ["[QUESTION]"]
        
        print("\n🎬 STARTING PIPELINE PROCESSING...")
        print("-----------------------------------")
        
        for video_file in VIDEO_QUEUE:
            print(f"Processing: {video_file}...", end=" ")
            
            result = self.process_video(video_file)
            
            if result:
                word, conf = result
                print(f"-> Predicted: '{word}' ({conf:.1%})")
                final_sentence.append(word)
            else:
                print("-> FAILED (Video Error)")

        print("-----------------------------------")
        print(f"\n>>> OUTPUT: {final_sentence}")

if __name__ == "__main__":
    app = PipelineDemo()
    app.run()