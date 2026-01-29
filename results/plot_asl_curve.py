import matplotlib.pyplot as plt
import re
import os

# --- CONFIGURATION ---
LOG_FILE = "asl_logs.txt"   # Ensure this file exists in the same folder!
OUTPUT_ACC = "asl_accuracy_curve.png"
OUTPUT_LOSS = "asl_loss_curve.png"

def parse_logs(filepath):
    epochs = []
    loss = []
    acc = []
    
    if not os.path.exists(filepath):
        print(f"❌ Error: Could not find '{filepath}'. Make sure you created it!")
        return [], [], []

    print(f"📖 Reading {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            # Looks for: "Epoch 1 Done. Loss: 2.5653 | Acc: 20.57%"
            match = re.search(r"Epoch (\d+) Done\. Loss: ([\d\.]+) \| Acc: ([\d\.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                loss.append(float(match.group(2)))
                acc.append(float(match.group(3)))
    
    print(f"✅ Found {len(epochs)} epochs of data.")
    return epochs, loss, acc

def plot_metric(epochs, values, title, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2, color=color, markersize=5)
    
    # Styling
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"🎨 Saved graph to {filename}")

if __name__ == "__main__":
    epochs, losses, accuracies = parse_logs(LOG_FILE)
    
    if epochs:
        # 1. Plot Accuracy
        plot_metric(epochs, accuracies, 
                   "ASL Model - Training Accuracy", 
                   "Accuracy (%)", 
                   OUTPUT_ACC, 
                   "tab:blue")
        
        # 2. Plot Loss
        plot_metric(epochs, losses, 
                   "ASL Model - Training Loss", 
                   "Cross Entropy Loss", 
                   OUTPUT_LOSS, 
                   "tab:red")
    else:
        print("⚠️ No data found. Please check your text file content.")