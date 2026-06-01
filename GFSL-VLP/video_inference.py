import os
import sys
import cv2
import yaml
import glob
import torch
import argparse
from torchvision import transforms
from train_slt import gloss_free_model, get_args_parser
from transformers import MBartTokenizer

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_architecture(checkpoint_path):
    print("📦 Booting the Translation Engine...")
    repo_parser = get_args_parser()
    repo_args, _ = repo_parser.parse_known_args([]) 
    
    yaml_files = glob.glob("configs/*.yaml")
    if not yaml_files:
         raise FileNotFoundError("Could not find the 'configs' folder or any .yaml files!")
         
    config_file = next((f for f in yaml_files if 'gloss_free' in f.lower()), yaml_files[0])
        
    print(f"📄 Reading architecture dimensions from: {config_file}")
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    model = gloss_free_model(config=config_dict, args=repo_args)

    print(f"📥 Unpacking weights safely into System RAM...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    del checkpoint
    torch.cuda.empty_cache()

    print("⚡ Sliding compiled model onto the RTX GPU...")
    model = model.to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.half() # Compress to FP16 to run smoothly on 4GB VRAM
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. Load the Model
    model = load_architecture(args.checkpoint)
    
    # 2. Load the mBART Vocabulary
    print("🗣️ Loading mBART Vocabulary...")
    tokenizer = MBartTokenizer.from_pretrained("./mbart_local", local_files_only=True)
    german_token_id = tokenizer.lang_code_to_id["de_DE"]

    # 3. Setup Video Transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    
    print("\n====================================================")
    print("📹 NEUROSIGN CONNECT - LIVE WEBCAM TRANSLATOR")
    print("• Press 'r' ONCE to START recording a sign.")
    print("• Press 's' ONCE to STOP and translate.")
    print("• Press 'q' to EXIT.")
    print("====================================================\n")

    frames = []
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        # If recording, show a red dot indicator and save frames
        if recording:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "RECORDING", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Instantly convert to tensor format while recording
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = transform(rgb_frame)
            frames.append(tensor_frame)
            
        cv2.imshow("NeuroSign Connect", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # START RECORDING
        if key == ord('r') and not recording:
            print("\n🔴 Recording started...")
            frames = [] # Clear the buffer
            recording = True
            
        # STOP RECORDING & TRANSLATE
        elif key == ord('s') and recording:
            print("⏹️ Recording stopped. Processing translation...")
            recording = False
            
            if len(frames) < 5:
                print("⚠️ Video too short! Please record for at least half a second.")
                continue
                
            # Stack into (Frames, Channels, Height, Width) - Keep it 4D!
            video_tensor = torch.stack(frames).to(DEVICE)
            
            if DEVICE.type == "cuda":
                video_tensor = video_tensor.half()
                
            print(f"📊 Tensor Shape: {video_tensor.shape}")
            
            # Package the video and frame count
            original_length = video_tensor.size(0)
            src_length = torch.tensor([original_length]).to(DEVICE)
            
            # [THE FIX]: Calculate the exact temporal shrinkage from the visual cortex!
            reduced_length = (((original_length - 4) // 2) - 4) // 2
            
            # Give mBART the mask that perfectly matches its new, smaller sequence
            attention_mask = torch.ones(1, reduced_length, dtype=torch.long).to(DEVICE)
            
            src_input = {
                'input_ids': video_tensor,
                'src_length_batch': src_length,
                'attention_mask': attention_mask
            }
            
            print("🧠 Running Model Inference...")
            try:
                with torch.no_grad():
                    output = model.generate(
                        src_input,                      
                        max_new_tokens=50,
                        num_beams=4,
                        decoder_start_token_id=german_token_id 
                    )
                    
                translation = tokenizer.decode(output[0], skip_special_tokens=True)
                print("\n====================================================")
                print("✅ TRANSLATION RESULT:")
                print(f"   {translation}")
                print("====================================================\n")
            except Exception as e:
                print(f"\n❌ Inference Error: {e}")
            
            # Clear memory so you can record another sentence safely
            del video_tensor, src_length, src_input
            if 'output' in locals():
                del output
            torch.cuda.empty_cache()
            
        # QUIT
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()