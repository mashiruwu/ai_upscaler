import os
import cv2
import numpy as np
from PIL import Image
import sys

# Add current directory to path so it can find upscaler.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from upscaler import Upscaler

def process_directory(directory, upscaler):
    print(f"\n--- Processing directory: {directory} ---")
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
        
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.heic']:
            continue
            
        # Skip files that are already upscaled outputs
        if filename.startswith("upscaled_"):
            continue
            
        # Check if an upscaled version already exists
        base_name = os.path.splitext(filename)[0]
        out_path_png = os.path.join(directory, f"upscaled_{base_name}.png")
        out_path_jpg = os.path.join(directory, f"upscaled_{base_name}.jpg")
        
        if os.path.exists(out_path_png) or os.path.exists(out_path_jpg):
            print(f"Skipping {filename} - already upscaled")
            continue
            
        input_path = os.path.join(directory, filename)
        print(f"Upscaling {filename}...")
        
        try:
            img = cv2.imread(input_path)
            if img is None:
                pil_img = Image.open(input_path).convert('RGB')
                img = np.array(pil_img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            img_normalized = img.astype(np.float32) / 255.0
            
            # Run the tiled inference
            upscaled = upscaler.tiled_inference(img_normalized, tile_size=400)
            
            # Save the result as JPG
            out_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path_jpg, out_bgr)
            print(f"Saved: {out_path_jpg}")
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    print("Loading model onto Apple Silicon (MPS)...")
    upscaler = Upscaler(device='mps')
    upscaler.load_model()
    
    dirs_to_process = [
        "/Users/diegohenrick/Downloads/lego-photos/AI-images/STAR_WARS",
        "/Users/diegohenrick/Downloads/lego-photos/AI-images/MARVEL",
        "/Users/diegohenrick/Downloads/lego-photos"
    ]
    
    for d in dirs_to_process:
        process_directory(d, upscaler)
        
    print("\nBatch processing complete!")
