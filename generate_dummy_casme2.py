"""
Generate Dummy CASME II Dataset
Generates fake facial frames and an Excel sheet that perfectly matches the CASME II 
format so the pipeline can be tested locally end-to-end.
"""
import os
import shutil
import numpy as np
import cv2
import pandas as pd

def generate_dummy_casme2(data_root="data/raw/CASME2", n_subjects=3, clips_per_sub=4, frames_per_clip=30):
    print("Generating synthetic CASME II dataset structure...")
    
    cropped_dir = os.path.join(data_root, "Cropped")
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    os.makedirs(cropped_dir, exist_ok=True)
    
    excel_data = []
    
    emotions = ["happiness", "disgust", "repression", "surprise", "fear", "sadness", "others"]
    
    for sub in range(1, n_subjects + 1):
        sub_name = f"sub{sub:02d}"
        sub_dir = os.path.join(cropped_dir, sub_name)
        os.makedirs(sub_dir, exist_ok=True)
        
        for clip in range(1, clips_per_sub + 1):
            clip_name = f"EP{sub:02d}_{clip:02d}f"
            clip_dir = os.path.join(sub_dir, clip_name)
            os.makedirs(clip_dir, exist_ok=True)
            
            # Generate frames
            # We'll just generate face-like frames consisting of colored noise
            for f in range(1, frames_per_clip + 1):
                img_path = os.path.join(clip_dir, f"img{f:04d}.jpg")
                # Gray face-ish color with random noise
                face = np.random.randint(100, 150, (112, 112, 3), dtype=np.uint8)
                cv2.imwrite(img_path, face)
                
            # Add annotation
            onset = 5
            offset = 25
            apex = 15
            emo = emotions[(sub * clip) % len(emotions)]
            
            excel_data.append({
                "Subject": sub_name,
                "Filename": clip_name,
                "Onset": onset,
                "Apex": apex,
                "Offset": offset,
                "Action Units": "4+12",
                "Estimated Emotion": emo
            })
            
    # Create excel
    df = pd.DataFrame(excel_data)
    excel_path = os.path.join(data_root, "CASME2-coding-20140508.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Dummy dataset generated at {data_root}")

if __name__ == "__main__":
    generate_dummy_casme2()
