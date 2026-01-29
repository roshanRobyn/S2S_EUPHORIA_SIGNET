import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import random

# --- PATH SETUP ---
INPUT_DIR = os.path.join("..", "data", "ASL_train_data") 
OUTPUT_DIR = os.path.join("..", "data", "processed")

# --- CONFIGURATION ---
TARGET_FRAMES = 64        
SAMPLES_PER_CLASS = 200   
CONFIDENCE_LEVEL = 0.3    # Lowered to 0.3 to catch difficult grayscale images

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE_LEVEL 
)

def process_letters():
    print(f"--- STARTING ROBUST PREPROCESSING ---")
    print(f"Input: {os.path.abspath(INPUT_DIR)}")
    print(f"Confidence Threshold: {CONFIDENCE_LEVEL}")

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Get Sorted Classes
    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    classes = [c for c in classes if c.upper() not in ['DEL', 'NOTHING', 'SPACE', 'J', 'Z']]
    classes.sort()

    if not classes:
        print("ERROR: No class folders found!")
        return

    print(f"Found {len(classes)} classes: {classes}")

    for letter in classes:
        class_dir = os.path.join(INPUT_DIR, letter)
        all_files = os.listdir(class_dir)
        
        # Filter for valid images first
        valid_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # --- SHUFFLE & LIMIT ---
        # Randomly pick 200 images to avoid getting a "bad batch" of consecutive frames
        random.shuffle(valid_files)
        target_files = valid_files[:SAMPLES_PER_CLASS]
        
        print(f"Processing '{letter}' (Attempting {len(target_files)} images)...")
        
        saved_count = 0
        failed_count = 0
        
        for filename in tqdm(target_files):
            filepath = os.path.join(class_dir, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                failed_count += 1
                continue
            
            # Grayscale -> RGB Conversion
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Detection
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Success! Extract Data
                frame_data = np.zeros((109, 3))
                hand_landmarks = results.multi_hand_landmarks[0]
                rh_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Center Normalize
                wrist = rh_points[0]
                rh_points = rh_points - wrist
                
                # Fill Slots
                frame_data[54:75] = rh_points
                
                # Save
                video_data = np.tile(frame_data, (TARGET_FRAMES, 1, 1))
                clean_name = os.path.splitext(filename)[0]
                save_name = f"LETTER_{letter}_{clean_name}.npy"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                np.save(save_path, video_data)
                
                saved_count += 1
            else:
                # Failure: MediaPipe saw no hand
                failed_count += 1
            
        # --- REPORT CARD ---
        print(f"  -> Result for '{letter}': {saved_count} Saved | {failed_count} Failed")
        
        if saved_count < 10:
            print(f"  ⚠️ WARNING: Very low detection rate for '{letter}'. Images might be too dark/blurry.")

    print(f"\nProcessing Complete! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    process_letters()