import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import STGCN
from features.facial_features import FACE_INDICES

# --- CONFIGURATION ---
# PUT YOUR TEST VIDEO HERE
INPUT_VIDEO_PATH = "15035.mp4" 

MODEL_PATH = os.path.join("..", "models", "stgcn_wlasl100_final.pth")
JSON_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
NUM_CLASSES = 100
MODEL_INPUT_SIZE = 64 # The exact size used in training

# --- LOAD LABELS ---
import json
with open(JSON_PATH, 'r') as f:
    data = json.load(f)[:NUM_CLASSES]
    labels = [entry['gloss'] for entry in data]

# --- HELPER: EXACT COPY OF TRAINING EXTRACTION ---
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
        
    return np.concatenate([pose, lh, rh, face])

def test_video_file():
    print(f"--- DEBUGGING VIDEO: {INPUT_VIDEO_PATH} ---")
    
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCN(num_class=NUM_CLASSES, in_channels=3, edge_strategy='spatial')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded.")
    else:
        print("CRITICAL ERROR: Model weights not found.")
        return

    model.to(device)
    model.eval()

    # 2. Process Video
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: {INPUT_VIDEO_PATH} does not exist.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    frames_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Extraction & Normalization
        keypoints = extract_landmarks(results)
        nose = keypoints[0]
        # Crucial Check: Only normalize if nose is actually found
        if np.any(nose): 
            keypoints = keypoints - nose
            
        frames_data.append(keypoints)

    cap.release()
    print(f"Extracted {len(frames_data)} frames from video.")

    if len(frames_data) == 0:
        print("Error: MediaPipe found no skeletons. Video might be too dark or person is hidden.")
        return

    # 3. PREPARE DATA (Exactly matching dataset.py)
    data_numpy = np.array(frames_data) # (T, 109, 3)
    T, V, C = data_numpy.shape
    
    # Pad/Crop to 64 frames
    final_input = np.zeros((MODEL_INPUT_SIZE, V, C))
    if T > MODEL_INPUT_SIZE:
        start = (T - MODEL_INPUT_SIZE) // 2
        final_input = data_numpy[start:start+MODEL_INPUT_SIZE, :, :]
    else:
        final_input[:T, :, :] = data_numpy

    # Reshape: (C, T, V, M)
    # 1. Transpose (T, V, C) -> (C, T, V)
    final_input = np.transpose(final_input, (2, 0, 1))
    # 2. Add Person Dimension -> (C, T, V, 1)
    final_input = np.expand_dims(final_input, axis=-1)
    # 3. Add Batch Dimension -> (1, C, T, V, 1)
    input_tensor = torch.from_numpy(final_input).float().unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # 4. PREDICT
    print("Running Prediction...")
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, 1)
        
        prediction = labels[idx.item()]
        confidence = conf.item() * 100

    print(f"\nRESULT:")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Debug: Show Top 3 guesses to see if the correct one is close
    print("\nTop 3 Candidates:")
    top3_prob, top3_idx = torch.topk(probs, 3)
    for i in range(3):
        p = top3_prob[0][i].item() * 100
        l = labels[top3_idx[0][i].item()]
        print(f"{i+1}. {l}: {p:.2f}%")

if __name__ == "__main__":
    test_video_file()