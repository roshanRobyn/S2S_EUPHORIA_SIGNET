import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import json
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import STGCN

# --- CONFIGURATION ---
DATA_DIR = os.path.join("..", "data", "processed_letters")
MODEL_SAVE_PATH = os.path.join("..", "models", "stgcn_letters_scratch.pth")
CLASS_SAVE_PATH = os.path.join("..", "data", "asl_classes.json")

BATCH_SIZE = 32          # Increased since samples are small
LEARNING_RATE = 0.01     # Higher starting LR for training from scratch
EPOCHS = 40              # Needs more epochs since starting from zero
NUM_FRAMES = 64          
SAMPLES_PER_CLASS = 200  # The Limit you requested

# --- 1. DATASET CLASS ---
class ASLDataset(Dataset):
    def __init__(self, data_dir, limit_per_class=None):
        self.data_dir = data_dir
        self.files = []
        self.labels = []
        
        # 1. Auto-detect classes
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"📂 Found {len(self.classes)} classes: {self.classes}")
        
        # 2. Save classes.json
        with open(CLASS_SAVE_PATH, 'w') as f:
            json.dump(self.classes, f)
        print(f"💾 Saved class list to {CLASS_SAVE_PATH}")

        # 3. Load File List with LIMIT
        total_loaded = 0
        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir, cls_name)
            npy_files = glob.glob(os.path.join(cls_folder, "*.npy"))
            
            # Shuffle to get a random 200, not just the first 200 (better variety)
            random.shuffle(npy_files)
            
            # Apply Limit
            if limit_per_class:
                npy_files = npy_files[:limit_per_class]
                
            for f in npy_files:
                self.files.append(f)
                self.labels.append(self.class_to_idx[cls_name])
            
            print(f"   -> Class '{cls_name}': Loaded {len(npy_files)} samples")
            total_loaded += len(npy_files)
                
        print(f"✅ Total Training Samples: {total_loaded}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load NPY (Shape: T, V, C)
        try:
            data = np.load(self.files[idx])
        except Exception as e:
            # Fallback for corrupt files
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(3, NUM_FRAMES, 109, 1), self.labels[idx]

        T, V, C = data.shape
        
        # Pad/Crop to NUM_FRAMES
        final_input = np.zeros((NUM_FRAMES, V, C))
        if T > NUM_FRAMES:
            start = (T - NUM_FRAMES) // 2
            final_input = data[start:start+NUM_FRAMES, :, :]
        else:
            final_input[:T, :, :] = data
            
        # Transpose (T, V, C) -> (C, T, V)
        data = np.transpose(final_input, (2, 0, 1))
        # Expand Person Dim -> (C, T, V, 1)
        data = np.expand_dims(data, axis=-1)
        
        return torch.FloatTensor(data), self.labels[idx]

# --- 2. TRAINING FUNCTION ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training FROM SCRATCH on {device}...")
    
    # 1. Prepare Data
    dataset = ASLDataset(DATA_DIR, limit_per_class=SAMPLES_PER_CLASS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    num_classes = len(dataset.classes)
    print(f"🧠 Initializing FRESH Model for {num_classes} classes...")

    # 2. Initialize Model (Fresh Weights)
    model = STGCN(num_class=num_classes, in_channels=3, edge_strategy='spatial')
    model.to(device)
    
    # 3. Optimizer (SGD is often better for STGCN from scratch, but Adam is safer for quick results)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Decay LR every 10 epochs
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        scheduler.step() # Lower learning rate over time
        
        avg_loss = total_loss / len(dataloader)
        curr_acc = 100.*correct/total
        print(f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f} | Acc: {curr_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save every time loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved Best Model -> {MODEL_SAVE_PATH}")

    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    train()