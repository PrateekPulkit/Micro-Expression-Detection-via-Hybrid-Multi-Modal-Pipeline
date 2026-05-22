import os
import urllib.request
from pathlib import Path

def download_mediapipe_models():
    """Download required MediaPipe Task models if they don't exist."""
    
    # Target directory relative to project root
    target_dir = Path("models/checkpoints")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    models = {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    }
    
    print("=== MediaPipe Model Setup ===")
    
    for filename, url in models.items():
        dest_path = target_dir / filename
        if dest_path.exists():
            print(f"[INFO] {filename} already exists. Skipping.")
        else:
            print(f"[INFO] Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
                print(f"[SUCCESS] Downloaded to {dest_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_mediapipe_models()
