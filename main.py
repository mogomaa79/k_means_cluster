#!/usr/bin/env python3
from pathlib import Path
from src.image_processor import process_image

def main():
    data_dir = Path('data')
    results_dir = Path('results')
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(data_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return

    for image_file in image_files:
        save_path = results_dir / f"{image_file.stem}.png"
        process_image(
            image_path=image_file, 
            save_path=save_path
        )

if __name__ == "__main__":
    main() 