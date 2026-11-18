# Satellite Image Building & Road Extraction with Deep Supervision

A research-oriented full-stack application for extracting **roads** and **building footprints** from high-resolution satellite imagery. It includes U-Net, Attention U-Net, deep-supervision training, a classical OpenCV baseline, evaluation metrics, a Flask API, and an interactive web dashboard.

> **Dataset policy:** datasets are not bundled in this ZIP. The project includes download/setup instructions for DeepGlobe Roads and SpaceNet 2 Buildings. You must review and comply with each dataset's license and citation terms.

## 1. Dataset plan

| Task | Dataset | Labels |
|---|---|---|
| Road extraction | [DeepGlobe Road Extraction](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) | Paired `_sat.jpg` images and `_mask.png` road masks |
| Building extraction | [SpaceNet 2 Buildings](https://spacenet.ai/spacenet-buildings-dataset-v2/) | Satellite image tiles and GeoJSON building footprints |

DeepGlobe provides 6,226 labelled 1024×1024 training images. The published validation and test sets do not include masks, so this project creates a reproducible train/validation split from the labelled training pairs. SpaceNet 2 building labels are polygons; the preparation script rasterizes GeoJSON footprints to binary masks.

**Important:** DeepGlobe is a road dataset, not a building dataset. SpaceNet 2 is used for buildings. Do not train a building model on DeepGlobe road masks.

## 2. Project structure

```text
satellite_extraction_project/
├── app.py
├── train.py
├── evaluate.py
├── run_classical.py
├── requirements.txt
├── .env.example
├── configs/default.json
├── satseg/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── inference.py
│   └── classical.py
├── scripts/
│   ├── download_deepglobe.py
│   └── prepare_spacenet.py
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## 3. Setup (Windows PowerShell / VS Code)

```powershell
# 1. Extract ZIP and open this folder in VS Code
cd satellite_extraction_project

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the CUDA version on your machine/Colab. See the official PyTorch installation selector: https://pytorch.org/get-started/locally/

## 4. Download datasets

### DeepGlobe Roads (Kaggle)

1. Create/login to a Kaggle account and accept the dataset's terms.
2. Configure Kaggle API credentials (`kaggle.json`) using Kaggle's account settings.
3. Run:

```powershell
python scripts/download_deepglobe.py
```

The script uses KaggleHub. If Kaggle authentication is required, configure your Kaggle credentials first. The download is several GB and is not included in this ZIP.

Expected data after download (the script prints the actual location):

```text
data/deepglobe/
  train/
    <id>_sat.jpg
    <id>_mask.png
```

If KaggleHub places the files in its cache, pass that folder to `--data-root` during training.

### SpaceNet 2 Buildings

SpaceNet 2 is a much larger dataset. The official SpaceNet page describes the AWS public bucket and individual AOI tarballs. For a manageable academic experiment, start with **AOI 3 Paris** (about 5.3 GB compressed training archive) or another AOI of your choice.

Install AWS CLI, then download an AOI archive (example: Paris):

```powershell
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz . --no-sign-request
```

Extract it under `data/spacenet2/`. The official data may be organized into image tiles and GeoJSON labels. The preparation utility below supports common SpaceNet GeoJSON + raster tile layouts; inspect the extracted layout and pass the relevant image/label directories.

```powershell
python scripts/prepare_spacenet.py --images data/spacenet2/images --labels data/spacenet2/geojson --output data/spacenet2/paired
```

The script requires GeoTIFF support (`rasterio`). It reports files it cannot pair. SpaceNet layouts can vary by release/AOI; if your archive differs, use the printed structure to adjust the paths or extend the pairing rules.

## 5. Train

Train a road model using DeepGlobe:

```powershell
python train.py --task roads --data-root "PATH_TO_DEEPGLOBE_TRAIN" --model attention_unet --epochs 20 --batch-size 4 --image-size 512
```

Train a building model using prepared SpaceNet pairs:

```powershell
python train.py --task buildings --data-root data/spacenet2/paired --model attention_unet --epochs 20 --batch-size 4 --image-size 512
```

Available models: `unet`, `attention_unet`. Deep supervision is enabled by default for both architectures. Use `--no-deep-supervision` to disable it. Checkpoints and logs are saved under `artifacts/<task>/<model>/`.

For Google Colab, upload/extract this ZIP into `/content`, install requirements, mount Google Drive if you want persistent data/checkpoints, and run the same commands with Colab paths. A GPU is recommended. Reduce batch size to 2 or image size to 384 if GPU memory is limited.

## 6. Evaluate

```powershell
python evaluate.py --task roads --data-root "PATH_TO_DEEPGLOBE_TRAIN" --checkpoint artifacts/roads/attention_unet/best.pt --model attention_unet
```

Metrics: pixel accuracy, precision, recall, F1/Dice, IoU. Results are saved as JSON. Do not report results until training/evaluation has actually completed.

## 7. Launch the application

```powershell
python app.py
```

Open `http://127.0.0.1:5000`. Upload a satellite image, choose Roads or Buildings, and select a trained model or the classical baseline. A neural model is only available after its checkpoint has been trained and placed in `artifacts/`.

## 8. Notes on scientific validity

- The road and building tasks use separate datasets and separate binary models.
- The current SpaceNet preparation script rasterizes vector footprints using the georeferencing of matching GeoTIFFs. Check alignment visually before training.
- Deep supervision applies auxiliary losses at decoder scales during training; inference uses the primary full-resolution output.
- The classical OpenCV pipeline is a baseline, not a learned model. It uses morphology and image contrast heuristics and should be evaluated on the same held-out split where labels are available.
- Use a fixed random seed and keep a held-out test set for final reporting.
- The application does not fabricate accuracy figures or claim a model is trained before a checkpoint exists.

## 9. Dataset citations

**DeepGlobe:** Demir, I. et al. (2018). *DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images*. CVPR Workshops.

**SpaceNet:** Van Etten, A., Lindenbaum, D., & Bacastow, T. M. (2018). *SpaceNet: A Remote Sensing Dataset and Challenge Series*. arXiv:1807.01232.

Please review dataset-specific terms before using or redistributing data. The ZIP contains code only, not the datasets.
## 10. Download Pretrained Models

TerraVision provides four pretrained deep learning models for satellite image building and road extraction.

You can download the trained model checkpoints from our GitHub Release:

**[Download All Pretrained Models – TerraVision v1.0.0](https://github.com/ursram-08/TerraVision-Satellite-Extraction/releases/tag/v1.0.0)**

### Available Models

| Model | Task | Architecture |
|---|---|---|
| best_unet.pth | Building extraction | U-Net |
| best_attention_unet.pth | Building extraction | Attention U-Net |
| best_road_unet.pth | Road extraction | U-Net |
| best_road_attention_unet.pth | Road extraction | Attention U-Net |

### Installation of Pretrained Models

1. Open the GitHub Release link above.
2. Download all four `.pth` files.
3. Open the `satellite_extraction_project` directory.
4. Locate or create a folder named `checkpoints`.
5. Place all four downloaded model files inside the `checkpoints` folder.
6. Start the Flask application using:

```powershell
python app.py
```

The application can then use the pretrained models for building and road extraction without requiring users to train them again.

**Note:** Dataset files and trained model weights are maintained separately from the source-code repository to keep the repository lightweight.