import json
import os
import cv2
import mediapipe as mp
import numpy as np
import sys

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from features.facial_features import FACE_INDICES
except ImportError:
    print("ERROR: Could not import FACE_INDICES.")
    print("Make sure 'features/facial_features.py' exists.")
    exit()

# --- CONFIGURATION ---
DATA_ROOT = os.path.join("..", "data")

# BACK TO BASICS: Use the main file
JSON_PATH = os.path.join(DATA_ROOT, "WLASL_v0.3.json") 

VIDEO_DIR = os.path.join(DATA_ROOT, "videos")        
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed")    

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- MEDIAPIPE SETUP ---
try:
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except AttributeError:
    # Fallback for newer Python versions
    print("WARNING: MediaPipe 'solutions' not found. Ensure mediapipe==0.10.9 is installed.")
    pass

def load_label_map(json_path, limit=100):
    """
    Reads the main WLASL JSON and keeps only the top N words.
    """
    if not os.path.exists(json_path):
        print(f"CRITICAL ERROR: Could not find {json_path}")
        exit()
        
    with open(json_path, 'r') as f:
        data = json.load(f)

    # SIMPLE LOGIC: Just take the first 'limit' items
    # WLASL is sorted by rank, so this gives us the most common words.
    print(f"Original dataset size: {len(data)} classes.")
    data = data[:limit]
    print(f"Sliced to top {limit} classes for training.")
    
    video_map = {}
    for entry in data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_map[video_id] = gloss
            
    return video_map

def extract_landmarks(results):
    # 1. Pose (33)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))
        
    # 2. Left Hand (21)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    else:
        lh = np.zeros((21, 3))

    # 3. Right Hand (21)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    else:
        rh = np.zeros((21, 3))

    # 4. Face (Subset)
    if results.face_landmarks:
        face_all = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
        face = face_all[FACE_INDICES] 
    else:
        face = np.zeros((len(FACE_INDICES), 3))
        
    return np.concatenate([pose, lh, rh, face])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        
        landmarks = extract_landmarks(results)
        
        # Normalization
        nose_xyz = landmarks[0] 
        if np.any(nose_xyz):
            landmarks = landmarks - nose_xyz
            
        frames_data.append(landmarks)
        
    cap.release()
    return np.array(frames_data)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"--- STARTING PREPROCESSING (LIMIT=100) ---")
    
    # 1. Load Map with HARD LIMIT
    video_map = load_label_map(JSON_PATH, limit=100)
    
    # 2. Check Videos
    if not os.path.exists(VIDEO_DIR):
        print(f"ERROR: Video folder not found at {VIDEO_DIR}")
        exit()

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    
    count = 0
    skipped = 0
    
    for video_file in video_files:
        video_id = video_file.split('.')[0]
        
        # Only process if this video belongs to our Top 100 words
        if video_id in video_map:
            input_path = os.path.join(VIDEO_DIR, video_file)
            output_path = os.path.join(OUTPUT_DIR, f"{video_id}.npy")
            
            if os.path.exists(output_path):
                continue
                
            try:
                data = process_video(input_path)
                if len(data) > 0:
                    np.save(output_path, data)
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} videos...")
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                skipped += 1

    print(f"--- COMPLETE ---")
    print(f"Processed: {count}")