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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import STGCN

# ==========================================
# ⚙️ ASL LETTERS CONFIGURATION
# ==========================================
MODE = "ASL"
DATA_DIR = os.path.join("..", "data", "processed_letters") # Folders A, B, C...
MODEL_PATH = os.path.join("..", "models", "stgcn_letters_scratch.pth")
CLASSES_PATH = os.path.join("..", "data", "asl_classes.json")
OUTPUT_IMAGE = "confusion_matrix_asl.png" # Saves to results/
BATCH_SIZE = 32
NUM_FRAMES = 64
# ==========================================

class EvaluationDataset(Dataset):
    def __init__(self, data_dir, classes):
        self.files = []
        self.labels = []
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        print(f"📂 Scanning ASL data from: {data_dir}")
        
        found_count = 0
        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_folder):
                files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.endswith('.npy')]
                
                # TEST SPLIT: Last 20%
                if len(files) > 0:
                    split_idx = int(len(files) * 0.8)
                    test_files = files[split_idx:]
                    if not test_files and files: test_files = [files[-1]]

                    for f in test_files:
                        self.files.append(f)
                        self.labels.append(self.class_to_idx[cls_name])
                    found_count += 1

        print(f"✅ Found data for {found_count}/{len(self.classes)} letters.")
        print(f"✅ Total Test Samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            T, V, C = data.shape
            
            final_input = np.zeros((NUM_FRAMES, V, C))
            if T > NUM_FRAMES:
                start = (T - NUM_FRAMES) // 2
                final_input = data[start:start+NUM_FRAMES, :, :]
            else:
                final_input[:T, :, :] = data
                
            data = np.transpose(final_input, (2, 0, 1))
            data = np.expand_dims(data, axis=-1)
            return torch.FloatTensor(data), self.labels[idx]
        except:
            return torch.zeros(3, NUM_FRAMES, 109, 1), 0

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting ASL Evaluation on {device}...")

    # 1. Load Classes
    if not os.path.exists(CLASSES_PATH):
        print(f"❌ Error: {CLASSES_PATH} not found. Did training finish?")
        return
        
    with open(CLASSES_PATH, 'r') as f:
        classes = json.load(f)
        
    # 2. Dataset
    dataset = EvaluationDataset(DATA_DIR, classes)
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    num_classes = len(classes)
    print(f"🧠 Loading Model ({num_classes} classes)...")
    model = STGCN(num_class=num_classes, in_channels=3, edge_strategy='spatial')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    else:
        print(f"❌ Model missing: {MODEL_PATH}")
        return

    # 4. Inference
    all_preds = []
    all_labels = []
    print("⏳ Running Inference...")
    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Metrics
    print("\n📊 ASL LETTERS RESULTS")
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    print("-" * 30)
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print("-" * 30)

    # 6. Plot
    print(f"🎨 Saving Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12)) # Bigger figure for all letters
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (ASL Letters)')
    
    save_path = os.path.join(os.path.dirname(__file__), OUTPUT_IMAGE)
    plt.savefig(save_path)
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    evaluate()