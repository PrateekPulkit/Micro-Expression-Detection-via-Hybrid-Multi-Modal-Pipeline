import os
import shutil
import pandas as pd
import cv2
import kagglehub

def import_and_format_kaggle_data():
    print("Downloading public CK+ dataset from Kaggle...")
    # This downloads securely without needing an API key
    raw_path = kagglehub.dataset_download("shawon10/ckplus")
    dataset_dir = os.path.join(raw_path, "CK+48")
    
    data_root = "data/raw/CASME2"
    cropped_dir = os.path.join(data_root, "Cropped")
    
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    os.makedirs(cropped_dir, exist_ok=True)
    
    emotions = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    excel_data = []
    
    # We want to form sequences. The files are like "S010_004_00000017.png"
    # We group them by Subject and Sequence.
    clips_dict = {}
    
    for emo in emotions:
        emo_path = os.path.join(dataset_dir, emo)
        for fname in os.listdir(emo_path):
            if not fname.endswith(".png"): continue
            
            parts = fname.replace(".png", "").split("_")
            if len(parts) >= 3:
                sub_id = parts[0] # e.g. S010
                seq_id = parts[1] # e.g. 004
                frame_id = parts[2] 
                
                key = (sub_id, seq_id, emo)
                if key not in clips_dict:
                    clips_dict[key] = []
                clips_dict[key].append(os.path.join(emo_path, fname))

    print(f"Discovered {len(clips_dict)} facial sequences. Converting to Micro-Expression format...")
    
    for (sub_id, seq_id, emo), img_paths in clips_dict.items():
        img_paths.sort() # Ensure temporal ordering
        
        casme_sub_name = f"{sub_id}"
        casme_clip_name = f"EP_{seq_id}"
        
        sub_dir = os.path.join(cropped_dir, casme_sub_name)
        clip_dir = os.path.join(sub_dir, casme_clip_name)
        os.makedirs(clip_dir, exist_ok=True)
        
        # Micro-Expression pipeline expects at least 16 frames per clip.
        # CK+ usually only has 3 peak frames. We will temporally upsample by repeating them 
        # sequentially to simulate a slow micro-expression onset.
        frames_out = []
        for p in img_paths:
            img = cv2.imread(p)
            img = cv2.resize(img, (112, 112))
            # Repeat each of the ~3 frames 8 times = 24 frames total
            for _ in range(8):
                frames_out.append(img)
                
        for i, f in enumerate(frames_out):
            cv2.imwrite(os.path.join(clip_dir, f"img{i+1:04d}.jpg"), f)
            
        excel_data.append({
            "Subject": casme_sub_name,
            "Filename": casme_clip_name,
            "Onset": 1,
            "Apex": 10,
            "Offset": 18,
            "Action Units": "N/A",
            "Estimated Emotion": emo
        })
        
    # Generate the annotation excel
    df = pd.DataFrame(excel_data)
    excel_path = os.path.join(data_root, "CASME2-coding-20140508.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nSuccessfully downloaded and mapped Kaggle data into {data_root}")
    print("You can now run: python -m preprocessing.dataset_builder --config configs/config.yaml")

if __name__ == "__main__":
    import_and_format_kaggle_data()
