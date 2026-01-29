# File: features/facial_features.py

# --- FACE MESH INDICES CONFIGURATION ---
# We define these here so they can be imported by both the 
# preprocessing script AND the live inference script.

# 1. Eyebrows (Grammar: Questions & Intensity)
# Indices for Left and Right eyebrows (approx 10 points)
EYEBROW_INDICES = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

# 2. Lips (Articulation: Mouthing words)
# Indices for Inner and Outer lip outlines (approx 20 points)
LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
               291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# 3. Eyelids (Emotion: Surprise vs. Anger)
# Top and bottom points of eyes to measure "openness"
EYE_OPENING_INDICES = [159, 145, 386, 374] 

# The Master List to extract
FACE_INDICES = EYEBROW_INDICES + LIP_INDICES + EYE_OPENING_INDICES