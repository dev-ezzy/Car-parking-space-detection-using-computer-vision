# prepare_data.py
import os
import argparse
from utils import get_parking_spots_from_mask, read_image, save_json

def main(mask_path='data/mask_1920_1080.png', images_dir='data/images', out_json='data/annotations.json'):
    slots = get_parking_spots_from_mask(mask_path)
    # create a list of frames with slot geometry
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    annotations = []
    for fname in files:
        annotations.append({
            "image": fname,
            "slots": [
                {"slot_id": s["slot_id"], "polygon": s["polygon"], "bbox": s["bbox"], "label": None}
                for s in slots
            ]
        })
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(annotations, out_json)
    print(f"Saved annotations for {len(files)} frames to {out_json}")

if __name__ == "__main__":
    main()
