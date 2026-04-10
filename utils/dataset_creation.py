from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Setup modèle
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Racine dataset
root = Path("dataset")
input_root = root / "images"
data_root = root / "data"
output_root = root / "output"

# Extensions supportées
exts = {".jpg", ".jpeg", ".png"}

# 🔢 1. Compter toutes les frames
all_images = [
    img_path
    for folder in input_root.iterdir() if folder.is_dir()
    for img_path in folder.iterdir()
    if img_path.suffix.lower() in exts
]

total_frames = len(all_images)
print(f"Total frames à traiter: {total_frames}")

# 🚀 2. Progress bar globale
with tqdm(total=total_frames, desc="Processing frames", unit="frame", smoothing=0.1) as pbar:

    for input_folder in input_root.iterdir():
        if not input_folder.is_dir():
            continue

        # Dossiers de sortie
        data_folder = data_root / input_folder.name
        output_folder = output_root / input_folder.name

        data_folder.mkdir(parents=True, exist_ok=True)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Images triées
        img_paths = sorted(
            [p for p in input_folder.iterdir() if p.suffix.lower() in exts]
        )

        for img_path in img_paths:
            base_name = img_path.stem
            npz_path = data_folder / f"{base_name}.npz"

            if npz_path.exists():
                pbar.update(1)
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"Erreur lecture: {img_path}")
                pbar.update(1)
                continue

            # Inference
            outputs = estimator.process_one_image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), inference_type="body"
            )

            base_name = img_path.stem

            # Save npz
            np.savez(data_folder / f"{base_name}.npz", *outputs)

            # Visualisation
            rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
            # cv2.imwrite(str(output_folder / f"{base_name}.jpg"),
            #             rend_img.astype(np.uint8))

            # Update barre
            pbar.update(1)