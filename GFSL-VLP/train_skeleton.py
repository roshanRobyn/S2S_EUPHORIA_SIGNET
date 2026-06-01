import os
import glob
import json
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MBartTokenizer
from models import gloss_free_model 
from train_slt import get_args_parser
import yaml

# --- 1. THE JSON LIBRARIAN ---
class OpenPoseDataset(Dataset):
    def __init__(self, csv_path, json_root_folder, tokenizer):
        print(f"📖 Reading Translations from: {csv_path}")
        try:
            self.metadata = pd.read_csv(csv_path, sep='\t')
            if 'SENTENCE_NAME' not in self.metadata.columns:
                self.metadata = pd.read_csv(csv_path) 
        except Exception as e:
            print(f"Error reading CSV: {e}")
            
        self.json_root = json_root_folder
        self.tokenizer = tokenizer
        
        self.valid_samples = []
        for index, row in self.metadata.iterrows():
            video_id = row['SENTENCE_NAME']
            folder_path = os.path.join(self.json_root, video_id)
            if os.path.exists(folder_path):
                self.valid_samples.append({
                    'video_id': video_id,
                    'text': row['SENTENCE']
                })
        print(f"✅ Found {len(self.valid_samples)} valid videos.")

    def __len__(self):
        return len(self.valid_samples)

    def parse_frame_json(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if len(data.get('people', [])) == 0:
            return [0.0] * 411 
            
        person = data['people'][0]
        keypoints = []
        keypoints.extend(person.get('pose_keypoints_2d', []))
        keypoints.extend(person.get('face_keypoints_2d', []))
        keypoints.extend(person.get('hand_left_keypoints_2d', []))
        keypoints.extend(person.get('hand_right_keypoints_2d', []))
        
        if len(keypoints) < 411:
            keypoints.extend([0.0] * (411 - len(keypoints)))
        return keypoints[:411]

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        folder_path = os.path.join(self.json_root, sample['video_id'])
        
        json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
        
        # 🛑 The Frame Guillotine: Prevent memory crash on giant videos
        json_files = json_files[:400]
        
        # 🛑 The Short Video Savior: Prevent BatchNorm crash on 1-frame videos
        if len(json_files) == 1:
            json_files.append(json_files[0]) # Duplicate the single frame
            
        frames_data = []
        for j_file in json_files:
            frames_data.append(self.parse_frame_json(j_file))
            
        # Fallback if the folder was completely empty (0 frames)
        if len(frames_data) == 0:
            frames_data = [[0.0] * 411, [0.0] * 411]
            
        video_tensor = torch.tensor(frames_data, dtype=torch.float32)
        
        labels = self.tokenizer(
            text_target=sample['text'], 
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).input_ids.squeeze()
            
        return {
            'input_ids': video_tensor,
            'labels': labels,
            'src_length_batch': torch.tensor([video_tensor.size(0)])
        }

# --- 2. THE TRAINING & VALIDATION ENGINE ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Engine on: {DEVICE}")

    # --- FILE PATHS ---
    TRAIN_CSV = "./data/train_bfh/how2sign_realigned_train.csv" 
    TRAIN_JSON = "./data/train_bfh/train/openpose_output/json/"
    
    VAL_CSV = "./data/train_bfh/how2sign_realigned_val.csv" 
    VAL_JSON = "./data/train_bfh/val/openpose_output/json/"
    
    tokenizer = MBartTokenizer.from_pretrained("./mbart_local", local_files_only=True, src_lang="en_XX", tgt_lang="en_XX")
    
    print("\n--- Loading Training Data ---")
    train_dataset = OpenPoseDataset(TRAIN_CSV, TRAIN_JSON, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    print("\n--- Loading Validation Data ---")
    val_dataset = OpenPoseDataset(VAL_CSV, VAL_JSON, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    repo_parser = get_args_parser()
    repo_args, _ = repo_parser.parse_known_args([]) 
    with open("configs/config_gloss_free.yaml", 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    model = gloss_free_model(config=config_dict, args=repo_args)
    checkpoint = torch.load("./mbart_local/pytorch_model.bin", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    model = model.to(DEVICE)
    
    print("\n🧊 Freezing mBART language weights...")
    for name, param in model.named_parameters():
        if "keypoint_projector" not in name and "sign_emb" not in name:
            param.requires_grad = False
            
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Lower the learning rate slightly to prevent future spikes
    optimizer = torch.optim.Adam(trainable_params, lr=0.00005)
    scaler = torch.amp.GradScaler('cuda')
    
    # NEW: Load your healthy Epoch 2 checkpoint to resume!
    print("\n💉 Injecting healthy Epoch 2 weights to cure NaN...")
    healthy_brain = torch.load("how2sign_epoch_2.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(healthy_brain, strict=False)
    del healthy_brain
    
    best_val_loss = float('inf') 
    
    # Change the loop to start at Epoch 3 since 1 and 2 are done!
    for epoch in range(3, 11):
        # === 1. TRAINING PHASE ===
        model.train() 
        total_train_loss = 0
        print(f"\n🔥 STARTING EPOCH {epoch} TRAINING...")
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True) 
            
            input_tensor = batch['input_ids'].to(DEVICE)
            attention_mask = torch.ones(input_tensor.shape[:2], dtype=torch.long).to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            tgt_attention_mask = torch.ones_like(labels)

            src_input = {'input_ids': input_tensor, 'src_length_batch': batch['src_length_batch'].to(DEVICE), 'attention_mask': attention_mask}
            
            with torch.amp.autocast('cuda'):
                logits = model(src_input, tgt_input={'input_ids': labels, 'attention_mask': tgt_attention_mask})
                loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            torch.cuda.empty_cache()
            
            if step % 50 == 0:
                print(f"Epoch: {epoch} | Train Video: {step}/{len(train_dataset)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # === 2. VALIDATION PHASE ===
        model.eval() # Pencils down!
        total_val_loss = 0
        print(f"\n📝 STARTING EPOCH {epoch} FINAL EXAM (VALIDATION)...")
        
        with torch.no_grad(): # Disable memory-heavy gradient tracking
            for step, batch in enumerate(val_dataloader):
                input_tensor = batch['input_ids'].to(DEVICE)
                attention_mask = torch.ones(input_tensor.shape[:2], dtype=torch.long).to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                tgt_attention_mask = torch.ones_like(labels)

                src_input = {'input_ids': input_tensor, 'src_length_batch': batch['src_length_batch'].to(DEVICE), 'attention_mask': attention_mask}
                
                with torch.amp.autocast('cuda'):
                    logits = model(src_input, tgt_input={'input_ids': labels, 'attention_mask': tgt_attention_mask})
                    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                total_val_loss += loss.item()
                torch.cuda.empty_cache()
                
        avg_val_loss = total_val_loss / len(val_dataset)
        print(f"📊 EPOCH {epoch} RESULTS -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # === 3. SAVE THE BEST BRAIN ===
        if avg_val_loss < best_val_loss:
            print(f"🏆 New Best Score! Saving best_model.pth...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_how2sign_model.pth")
        
        torch.save(model.state_dict(), f"how2sign_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()