import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
import random

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, json_path, frames=64, mode='train'):
        self.data_dir = data_dir
        self.frames = frames
        self.mode = mode
        
        self.samples = [] # List of (filename, label_index)
        self.labels = []  # List of string names ["book", "drink", ..., "A", "B"]
        
        # --- 1. LOAD WLASL WORDS (Indices 0-99) ---
        try:
            with open(json_path, 'r') as f:
                # Load only first 100 words
                raw_data = json.load(f)[:100] 
                
            for idx, entry in enumerate(raw_data):
                word = entry['gloss']
                self.labels.append(word)
                
                # Add all video instances for this word
                for inst in entry['instances']:
                    # Only look for processed .npy files
                    video_id = inst['video_id']
                    path = os.path.join(data_dir, f"{video_id}.npy")
                    
                    if os.path.exists(path):
                        self.samples.append((path, idx))
                        
            print(f"Loaded {len(self.labels)} Word Classes.")
            
        except FileNotFoundError:
            print(f"CRITICAL WARNING: WLASL JSON not found at {json_path}")

        # --- 2. LOAD ASL LETTERS (Indices 100-123) ---
        # Scan folder for "LETTER_A_...", "LETTER_B_..."
        letter_map = {} # Maps "A" -> 100, "B" -> 101...
        
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            letter_files = [f for f in files if f.startswith("LETTER_")]
            
            for f in letter_files:
                # Filename format: LETTER_A_filename.npy
                parts = f.split('_')
                if len(parts) < 3: continue
                
                letter_name = parts[1] # "A"
                
                # If we haven't seen this letter yet, add it to labels
                if letter_name not in self.labels:
                    self.labels.append(letter_name)
                
                # Get the index (e.g., "A" might be index 100)
                label_idx = self.labels.index(letter_name)
                
                full_path = os.path.join(data_dir, f)
                self.samples.append((full_path, label_idx))
                
        print(f"Total Classes: {len(self.labels)} (Words + Letters)")
        print(f"Total Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load Data
        data = np.load(path) # Shape (Frames, 109, 3) or similar
        
        # --- DYNAMIC PADDING/SAMPLING ---
        # Ensure exactly 64 frames
        T, V, C = data.shape
        if T < self.frames:
            # Pad with zeros
            pad = np.zeros((self.frames - T, V, C))
            data = np.concatenate([data, pad], axis=0)
        elif T > self.frames:
            # Random crop
            start = random.randint(0, T - self.frames)
            data = data[start:start+self.frames]
            
        # --- AUGMENTATION: THE "BODY DROPOUT" ---
        # If this is a Word (label < 100), sometimes delete the body
        # to match the "Hands Only" style of the letters.
        if self.mode == 'train' and label < 100:
            if random.random() < 0.3: # 30% chance
                # WLASL Structure: [Pose(33), L_Hand(21), R_Hand(21), Face(34)]
                # Keep only hands (Indices 33 to 75)
                # Zero out Pose (0-33) and Face (75-109)
                temp = data.copy()
                temp[:, :33, :] = 0   # Kill Body
                temp[:, 75:, :] = 0   # Kill Face
                data = temp

        # Convert to Tensor (Channel First for ST-GCN)
        # (Frames, Nodes, Channels) -> (Channels, Frames, Nodes)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1) 
        
        # Add "Person" dimension (1 person) -> (C, T, V, M)
        data = data.unsqueeze(-1)

        return data, label
