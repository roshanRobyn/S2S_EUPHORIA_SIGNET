import os
import shutil
import json
import operator

# --- CONFIGURATION ---
JSON_PATH = "WLASL_v0.3.json"          # Path to your master JSON file
SOURCE_FOLDER = "processed"       # Where your flat .npy files are
DEST_FOLDER = "processed_data_sorted"  # Where organized folders go
TARGET_COUNT = 100                     # STRICTLY limit to Top 100 classes

def organize_strict():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    print(f"📖 Reading {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 1. IDENTIFY TOP 100 GLOSSES (Standard WLASL Protocol)
    # We count how many instances each gloss has
    gloss_counts = {}
    for entry in data:
        gloss = entry['gloss']
        count = len(entry['instances'])
        gloss_counts[gloss] = count
    
    # Sort by count (descending) and take top 100
    sorted_glosses = sorted(gloss_counts.items(), key=operator.itemgetter(1), reverse=True)
    top_100_glosses = {g[0] for g in sorted_glosses[:TARGET_COUNT]}
    
    print(f"✅ Identified Top {TARGET_COUNT} glosses (e.g., {list(top_100_glosses)[:5]}...)")

    # 2. MAP VIDEO IDs -> GLOSS (Only for Top 100)
    id_to_gloss = {}
    for entry in data:
        gloss = entry['gloss']
        if gloss in top_100_glosses: # <--- CRITICAL FILTER
            for instance in entry['instances']:
                video_id = instance['video_id']
                id_to_gloss[video_id] = gloss

    print(f"✅ Loaded {len(id_to_gloss)} valid video IDs for WLASL-100.")

    # 3. MOVE FILES
    print(f"📂 Scanning '{SOURCE_FOLDER}'...")
    os.makedirs(DEST_FOLDER, exist_ok=True)
    
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".npy")]
    count = 0
    skipped_letters = 0
    skipped_rare = 0

    for filename in files:
        # Check if file is a WLASL ID or a Letter
        video_id = os.path.splitext(filename)[0]

        if video_id in id_to_gloss:
            # It is a VALID Top-100 WLASL file
            gloss = id_to_gloss[video_id]
            target_dir = os.path.join(DEST_FOLDER, gloss)
            os.makedirs(target_dir, exist_ok=True)
            
            src = os.path.join(SOURCE_FOLDER, filename)
            dst = os.path.join(target_dir, filename)
            shutil.copy(src, dst) # Using COPY to be safe (keeps original)
            count += 1
        
        elif "A" <= video_id[0] <= "Z" and "_" in video_id: 
            # Likely a letter file like "A_1.npy"
            skipped_letters += 1
        else:
            # Likely a WLASL file that isn't in Top 100
            skipped_rare += 1

    print("-" * 30)
    print(f"🎉 Processed {len(files)} files:")
    print(f"   ✅ Moved {count} files to '{DEST_FOLDER}' (Strict WLASL-100)")
    print(f"   🙈 Skipped {skipped_letters} letter files (e.g. A_1.npy)")
    print(f"   🙈 Skipped {skipped_rare} rare WLASL files (not in Top 100)")
    print("-" * 30)
    
    print(f"🚀 Now run 'evaluation/evaluate_wlasl.py'. Point DATA_DIR to '{DEST_FOLDER}'!")

if __name__ == "__main__":
    organize_strict()