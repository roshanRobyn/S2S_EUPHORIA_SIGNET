import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import SignLanguageDataset
from models.model import STGCN
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build the correct paths automatically
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
JSON_PATH = os.path.join(PROJECT_ROOT, "data", "WLASL_v0.3.json")

BATCH_SIZE = 16
EPOCHS = 40  # Increased slightly for the new data
LR = 0.01    # Lower learning rate for stability

def train():
    # 1. Setup Data
    print("--- Loading Unified Dataset (Words + Letters) ---")
    full_dataset = SignLanguageDataset(DATA_DIR, JSON_PATH, mode='train')
    
    # Calculate Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    num_classes = len(full_dataset.labels) # Should be ~124
    print(f"Model configured for {num_classes} classes.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = STGCN(num_class=num_classes, in_channels=3, edge_strategy='spatial')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 3. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            # New Line (Goes up one folder level first)
            save_path = os.path.join("..", "models", "stgcn_unified_final.pth")
            torch.save(model.state_dict(), save_path)
            print("  --> Model Saved!")

    print("Training Complete!")

if __name__ == "__main__":
    train()