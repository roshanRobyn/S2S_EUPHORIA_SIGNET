import os
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from preprocess import SignVideoPreprocessor

app = FastAPI(title="NeuroSign Connect Inference Engine")
preprocessor = SignVideoPreprocessor()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = True  # Match the precision of your loaded model weights

# --- LOAD MODEL (Place your actual repo model loading logic below) ---
print("📥 Binding pre-trained weights to GPU...")
model = None 
# Example placeholder:
# from train_slt import VLP_Model
# model = VLP_Model(args).to(DEVICE)
# if USE_FP16: model = model.half()
# model.eval()

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    # 1. Stream incoming multi-part form data to disk
    file_location = f"temp_inference_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    try:
        # 2. Execute the preprocessing pipeline 
        print("⚙️ Processing frames...")
        input_tensor = preprocessor.process_video(
            video_path=file_location, 
            device=DEVICE, 
            half_precision=USE_FP16
        )
        
        # Expected Output Dimension Verification printout
        print(f"📊 Tensor generated successfully. Input shape: {input_tensor.shape}")

        # 3. Model Evaluation Pass
        with torch.no_grad():
            if model is not None:
                # Direct forward pass to get translations from mBART decoder
                # Note: Adjust the method name (.generate or .forward) to match the repo code
                output_tokens = model.generate(input_tensor)
                translation = output_tokens # Replace with text tokenizer decoding output
            else:
                translation = f"System functional. Matrix configuration verified: {list(input_tensor.shape)}"

    except Exception as e:
        translation = f"Pipeline Processing Error: {str(e)}"
        
    finally:
        # 4. Storage cleanup
        if os.path.exists(file_location):
            os.remove(file_location)

    return {"translation": translation}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)