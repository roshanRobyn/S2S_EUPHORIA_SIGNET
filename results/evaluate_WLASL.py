import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- PATH SETUP ---
# Appends project root so we can import 'models'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import STGCN

# ==========================================
# ⚙️ WLASL CONFIGURATION
# ==========================================
MODE = "WLASL"

# UPDATED: Points to the sorted folder you just created
DATA_DIR = os.path.join("..", "data", "processed_data_sorted") 

MODEL_PATH = os.path.join("..", "models", "stgcn_wlasl100_final.pth")
CLASSES_PATH = os.path.join("..", "data", "WLASL_v0.3.json")
NUM_CLASSES = 100 
OUTPUT_IMAGE = "confusion_matrix_wlasl.png" # Saves in current folder (results/)
BATCH_SIZE = 32
NUM_FRAMES = 64
# ==========================================

class EvaluationDataset(Dataset):
    def __init__(self, data_dir, classes):
        self.files = []
        self.labels = []
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        print(f"📂 Scanning WLASL data from: {data_dir}")
        
        found_classes = 0
        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir, cls_name)
            
            if os.path.isdir(cls_folder):
                found_classes += 1
                files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.endswith('.npy')]
                
                # SIMULATED TEST SPLIT: Take the last 20%
                if len(files) > 0:
                    split_idx = int(len(files) * 0.8)
                    test_files = files[split_idx:]
                    
                    # Safety: If <5 files, split_idx might be same as len, so take at least the last one
                    if not test_files and files: test_files = [files[-1]]

                    for f in test_files:
                        self.files.append(f)
                        self.labels.append(self.class_to_idx[cls_name])
            else:
                pass 

        print(f"✅ Found data for {found_classes}/{len(self.classes)} classes.")
        print(f"✅ Total Test Samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            T, V, C = data.shape
            
            # Pad/Crop to 64
            final_input = np.zeros((NUM_FRAMES, V, C))
            if T > NUM_FRAMES:
                start = (T - NUM_FRAMES) // 2
                final_input = data[start:start+NUM_FRAMES, :, :]
            else:
                final_input[:T, :, :] = data
                
            data = np.transpose(final_input, (2, 0, 1))
            data = np.expand_dims(data, axis=-1)
            
            return torch.FloatTensor(data), self.labels[idx]
        except Exception as e:
            return torch.zeros(3, NUM_FRAMES, 109, 1), 0

def load_classes(path):
    with open(path, 'r') as f:
        data = json.load(f)
        # Extract first 100 glosses (Standard WLASL-100 logic)
        # We sort by instance count to match the strict organizer
        if isinstance(data, list) and isinstance(data[0], dict) and 'gloss' in data[0]:
             # Sort by most frequent (Top 100)
             sorted_data = sorted(data, key=lambda x: len(x['instances']), reverse=True)
             return [d['gloss'] for d in sorted_data][:NUM_CLASSES]
        return data

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting WLASL Evaluation on {device}...")

    # 1. Load Classes
    classes = load_classes(CLASSES_PATH)
    
    # 2. Check Data
    if not os.path.exists(DATA_DIR):
         print(f"❌ Error: Folder '{DATA_DIR}' not found.")
         print("   Did you run 'data/organize_wlasl100_strict.py'?")
         return

    dataset = EvaluationDataset(DATA_DIR, classes)
    if len(dataset) == 0:
        print("❌ No data found in the folders. Check your paths.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load Model
    print("🧠 Loading Model...")
    model = STGCN(num_class=NUM_CLASSES, in_channels=3, edge_strategy='spatial')
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    else:
        print(f"❌ Model file missing: {MODEL_PATH}")
        return

    # 4. Inference
    all_preds = []
    all_labels = []

    print("⏳ Running Inference on Test Set...")
    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Metrics
    print("\n📊 WLASL RESULTS")
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Weighted avg is fairer for imbalanced WLASL
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print("-" * 30)
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print("-" * 30)

    # 6. Plot (Top 20 classes)
    print(f"🎨 Saving Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    subset_classes = classes[:20]
    cm_subset = cm[:20, :20]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Greens', xticklabels=subset_classes, yticklabels=subset_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Top 20 WLASL Classes)')
    
    # Save to RESULTS folder
    save_path = os.path.join(os.path.dirname(__file__), OUTPUT_IMAGE)
    plt.savefig(save_path)
    print(f"Done! Image saved to {save_path}")

if __name__ == "__main__":
    evaluate()