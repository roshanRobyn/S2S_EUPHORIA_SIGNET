
import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
import json
import asyncio

# --- PATH SETUP for SLM Import ---
# We need to see the 'models' folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.slm_model2 import SLMClient

# --- CONFIGURATION ---
# 1. VIDEO INPUTS
VIDEO_QUEUE = [
    "what.mp4",
    "eat.mp4", 
    "now.mp4"
]

# 2. MODEL PATHS
STGCN_MODEL_PATH = os.path.join("..", "models", "stgcn_wlasl100_final.pth") 
CLASSES_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
MODEL_INPUT_SIZE = 64

# --- STAGE 1: THE EYES (Video -> Gloss) ---
class VideoRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"👀 Initializing Vision Model on {self.device}...")
        
        self.load_classes()
        self.load_model()
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Placeholder for Face Indices (Using standard subset logic implicitly via slicing if needed)
        # Ideally this should match your training exactly.
        self.FACE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

    def load_classes(self):
        try:
            with open(CLASSES_PATH, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and isinstance(data[0], dict):
                     self.labels = [entry['gloss'] for entry in data]
                else:
                     self.labels = data
                # Strict cutoff to match model
                self.labels = self.labels[:100] 
        except Exception as e:
            print(f"❌ Error loading classes: {e}")
            sys.exit()

    def load_model(self):
        # Local Import
        from models.model import STGCN 
        self.model = STGCN(num_class=100, in_channels=3, edge_strategy='spatial')
        if os.path.exists(STGCN_MODEL_PATH):
            self.model.load_state_dict(torch.load(STGCN_MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            print(f"❌ STGCN Model not found at {STGCN_MODEL_PATH}")
            sys.exit()

    def extract_keypoints(self, results):
        if results.pose_landmarks: pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        else: pose = np.zeros((33, 3))
        
        if results.left_hand_landmarks: lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        else: lh = np.zeros((21, 3))
        
        if results.right_hand_landmarks: rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        else: rh = np.zeros((21, 3))
        
        if results.face_landmarks:
            all_face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
            # Slicing to ensure we fit the 109 node limit (33+21+21 = 75 used, leaving 34 for face)
            face = all_face[:34] 
        else: 
            face = np.zeros((34, 3))
            
        keypoints = np.concatenate([pose, lh, rh, face])
        
        # Normalize (Subtract Nose) - Matches your working script
        nose = keypoints[0]
        if np.any(nose): keypoints = keypoints - nose
            
        return keypoints

    def process_video_queue(self, video_files):
        gloss_sequence = []
        print("\n🎥 PROCESSING VIDEOS...")
        
        for video_path in video_files:
            if not os.path.exists(video_path):
                print(f"   ⚠️ Skipping missing file: {video_path}")
                continue

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

            if not frames_data: continue

            # Pre-process (Pad/Crop to 64)
            data_numpy = np.array(frames_data)
            T, V, C = data_numpy.shape
            final_input = np.zeros((MODEL_INPUT_SIZE, V, C))
            if T > MODEL_INPUT_SIZE:
                start = (T - MODEL_INPUT_SIZE) // 2
                final_input = data_numpy[start:start+MODEL_INPUT_SIZE, :, :]
            else:
                final_input[:T, :, :] = data_numpy
            
            # Transpose to (1, C, T, V, 1)
            final_input = np.transpose(final_input, (2, 0, 1))
            final_input = np.expand_dims(final_input, axis=-1)
            inp_tensor = torch.from_numpy(final_input).float().unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                out = self.model(inp_tensor)
                prob = torch.softmax(out, dim=1)
                conf, idx = torch.max(prob, 1)
                word = self.labels[idx.item()]
                print(f"   ✅ Video '{video_path}' -> GLOSS: '{word}' ({conf.item():.1%})")
                gloss_sequence.append(word)
                
        return gloss_sequence

# --- STAGE 2: THE BRAIN (Gloss -> English) ---
async def run_slm_translation(glosses):
    print("\n🧠 INITIALIZING LANGUAGE MODEL (SLM)...")
    
    # Init Client (Downloads Phi-2 if needed)
    client = SLMClient()
    
    # Context prompt (NMM)
    nmm = ["[QUESTION]"]
    
    print(f"   📥 Input Glosses: {glosses}")
    print("   ⏳ Generating Natural English...")
    
    full_response = ""
    print("\n   💬 TRANSLATION: ", end="", flush=True)
    
    # Stream the response
    async for word in client.stream_translate(glosses, nmm):
        print(word, end=" ", flush=True)
        full_response += word + " "
        
    print("\n")
    return full_response.strip()

# --- MAIN EXECUTION ---
async def main():
    print("=========================================")
    print("      SIGN LANGUAGE TRANSLATION DEMO     ")
    print("=========================================")
    
    # 1. Run Vision
    vision_system = VideoRecognizer()
    glosses = vision_system.process_video_queue(VIDEO_QUEUE)
    
    if not glosses:
        print("❌ No glosses detected. Aborting SLM stage.")
        return

    # 2. Run Language
    english_output = await run_slm_translation(glosses)
    
    print("=========================================")
    print(f"🎉 FINAL RESULT: {english_output}")
    print("=========================================")

if __name__ == "__main__":
    asyncio.run(main())