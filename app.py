import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

from satseg.classical import classical_extract
from satseg.inference import predict_image


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

UPLOAD = Path("uploads")
RESULTS = Path("static/results")
CHECKPOINTS = Path("checkpoints")

UPLOAD.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


MODEL_FILES = {
    ("buildings", "unet"): "best_unet.pth",
    ("buildings", "attention_unet"): "best_attention_unet.pth",
    ("roads", "unet"): "best_road_unet.pth",
    ("roads", "attention_unet"): "best_road_attention_unet.pth",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    models = []

    for (task, model), filename in MODEL_FILES.items():
        checkpoint = CHECKPOINTS / filename

        models.append({
            "task": task,
            "model": model,
            "ready": checkpoint.exists()
        })

    return jsonify({
        "api": "online",
        "trained_models": models
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify(error="Please select an image."), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify(error="The selected file has no name."), 400

    task = request.form.get("task", "roads")
    model = request.form.get("model", "classical")

    if task not in ("roads", "buildings"):
        return jsonify(error="Invalid task."), 400

    if model not in ("classical", "unet", "attention_unet"):
        return jsonify(error="Invalid model."), 400

    extension = Path(file.filename).suffix.lower()

    if extension not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return jsonify(error="Use PNG, JPG or TIFF."), 400

    uid = uuid.uuid4().hex
    image_path = UPLOAD / (uid + extension)

    file.save(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)

        if model == "classical":

            mask = classical_extract(image_array, task)
            method = "Classical morphology baseline"

        else:

            checkpoint = CHECKPOINTS / MODEL_FILES[(task, model)]

            if not checkpoint.exists():
                return jsonify(
                    error=f"Checkpoint not found: {checkpoint}"
                ), 400

            _, mask_image = predict_image(
                image_path=image_path,
                checkpoint=checkpoint,
                model_name=model,
                task=task
            )

            mask = np.asarray(mask_image)
            method = model.replace("_", " ").title()

        mask_name = uid + "_mask.png"
        Image.fromarray(mask).save(RESULTS / mask_name)

        overlay = image_array.copy()
        binary = mask > 0

        overlay[binary] = (
            0.35 * overlay[binary] +
            0.65 * np.array([30, 220, 255])
        ).astype("uint8")

        overlay_name = uid + "_overlay.png"
        Image.fromarray(overlay).save(RESULTS / overlay_name)

        return jsonify(
            original="/" + str(image_path).replace("\\", "/"),
            mask="/static/results/" + mask_name,
            overlay="/static/results/" + overlay_name,
            method=method,
            task=task
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify(error=str(error)), 500

    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )