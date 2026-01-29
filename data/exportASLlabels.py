import sys
import os
import json

# Point to project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import SignLanguageDataset

# Configuration
DATA_DIR = os.path.join("..", "data", "processed")
JSON_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
OUTPUT_FILE = "classes.json"

def export():
    print("--- RECONSTRUCTING LABEL ORDER ---")
    # We initialize the dataset exactly like the training script did
    # This ensures the logic (and order) is identical.
    dataset = SignLanguageDataset(DATA_DIR, JSON_PATH, mode='train')
    
    labels = dataset.labels
    print(f"\n✅ Extracted {len(labels)} classes.")
    print(f"First 5: {labels[:5]}")
    print(f"Last 5:  {labels[-5:]}")
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(labels, f, indent=4)
    
    print(f"\nSaved to {os.path.abspath(OUTPUT_FILE)}")
    print("Use this file for your Realtime Demo!")

if __name__ == "__main__":
    export()