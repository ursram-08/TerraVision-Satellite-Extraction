from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import build_model


# Trained checkpoint filenames
CHECKPOINTS = {
    ("buildings", "unet"): "best_unet.pth",
    ("buildings", "attention_unet"): "best_attention_unet.pth",
    ("roads", "unet"): "best_road_unet.pth",
    ("roads", "attention_unet"): "best_road_attention_unet.pth",
}

# Original training image dimensions
TILE_SIZES = {
    "buildings": 650,
    "roads": 1024,
}

MODEL_INPUT_SIZE = 256
OVERLAP_RATIO = 0.20


def load_checkpoint_model(checkpoint, model_name, task, device):
    model = build_model(model_name, task=task).to(device)

    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True
    )

    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]

    model.load_state_dict(payload, strict=True)
    model.eval()

    return model


def preprocess_image(image, task, image_size=256):
    resized = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR
    )

    x = np.asarray(resized, dtype=np.float32)

    if task == "roads":
        x = x / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        x = (x - mean) / std

    elif task == "buildings":
        valid = np.any(x != 0, axis=2)

        if not np.any(valid):
            valid = np.ones(x.shape[:2], dtype=bool)

        for channel in range(3):
            values = x[:, :, channel][valid]

            low, high = np.percentile(values, [2, 98])

            scale = max(float(high - low), 1.0)

            x[:, :, channel] = np.clip(
                (x[:, :, channel] - low) / scale,
                0.0,
                1.0
            )

    else:
        raise ValueError(f"Unsupported task: {task}")

    x = torch.from_numpy(x.copy())

    x = x.permute(2, 0, 1).unsqueeze(0)

    return x


def get_tile_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap

    positions = list(range(0, length - tile_size + 1, stride))

    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


@torch.no_grad()
def predict_image(
    image_path,
    checkpoint,
    model_name="unet",
    task="roads",
    threshold=0.5
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    original = Image.open(image_path).convert("RGB")

    image_array = np.asarray(original)

    height, width = image_array.shape[:2]

    tile_size = TILE_SIZES[task]

    overlap = int(tile_size * OVERLAP_RATIO)

    y_positions = get_tile_positions(
        height,
        tile_size,
        overlap
    )

    x_positions = get_tile_positions(
        width,
        tile_size,
        overlap
    )

    model = load_checkpoint_model(
        checkpoint=checkpoint,
        model_name=model_name,
        task=task,
        device=device
    )

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for y in y_positions:
        for x in x_positions:

            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            tile_array = image_array[y:y_end, x:x_end]

            tile_image = Image.fromarray(tile_array)

            input_tensor = preprocess_image(
                tile_image,
                task,
                image_size=MODEL_INPUT_SIZE
            ).to(device)

            outputs = model(input_tensor)

            logits = (
                outputs[0]
                if isinstance(outputs, tuple)
                else outputs
            )

            probabilities = torch.sigmoid(logits)

            # Restore prediction to original tile dimensions
            probabilities = F.interpolate(
                probabilities,
                size=tile_array.shape[:2],
                mode="bilinear",
                align_corners=False
            )

            probabilities = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            probability_sum[y:y_end, x:x_end] += probabilities

            prediction_count[y:y_end, x:x_end] += 1.0

    # Average overlapping tile predictions
    final_probability = (
        probability_sum /
        np.maximum(prediction_count, 1.0)
    )

    final_mask = (
        final_probability >= threshold
    ).astype(np.uint8) * 255

    mask_image = Image.fromarray(final_mask)

    return original, mask_image