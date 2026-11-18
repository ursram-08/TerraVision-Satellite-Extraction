from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

class PairedSegmentationDataset(Dataset):
    """Pairs images with masks by stem; accepts *_sat.jpg + *_mask.png or images/ and masks/."""
    def __init__(self, root, image_size=512, task="roads"):
        self.root = Path(root)
        self.image_size = image_size
        self.task = task
        pairs = []
        # DeepGlobe convention
        for img in self.root.rglob("*_sat.jpg"):
            mask = img.with_name(img.name.replace("_sat.jpg", "_mask.png"))
            if mask.exists(): pairs.append((img, mask))
        # Generic paired folders / SpaceNet prepared layout
        if not pairs:
            image_dir = self.root / "images"
            mask_dir = self.root / "masks"
            if image_dir.exists() and mask_dir.exists():
                masks = {p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in IMG_EXTS}
                for img in image_dir.rglob("*"):
                    if img.suffix.lower() in IMG_EXTS and img.stem in masks:
                        pairs.append((img, masks[img.stem]))
        # Prepared flat pair layout: same basename image and _mask.png
        if not pairs:
            for img in self.root.rglob("*"):
                if img.suffix.lower() in IMG_EXTS and not any(t in img.stem.lower() for t in ("mask", "label")):
                    candidates = [img.with_name(img.stem + "_mask.png"), img.with_name(img.stem + ".png")]
                    mask = next((m for m in candidates if m.exists() and m != img), None)
                    if mask: pairs.append((img, mask))
        self.pairs = sorted(set(pairs))
        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs found under {self.root}. See README for expected structure.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        image = Image.open(ip).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(mp).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y

def make_loaders(root, image_size=512, batch_size=4, val_fraction=.2, workers=2, seed=42):
    ds = PairedSegmentationDataset(root, image_size=image_size)
    n_val = max(1, int(len(ds) * val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available()),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available()),
        len(ds)
    )
