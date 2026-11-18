from pathlib import Path
import argparse, shutil
import kagglehub

p=argparse.ArgumentParser()
p.add_argument("--output",default="data/deepglobe")
a=p.parse_args()
dest=Path(a.output); dest.mkdir(parents=True,exist_ok=True)
print("Downloading DeepGlobe Road Extraction dataset from Kaggle...")
src=Path(kagglehub.dataset_download("balraj98/deepglobe-road-extraction-dataset"))
print(f"KaggleHub dataset cache: {src}")
print("Dataset is downloaded. Use the cache path as --data-root if it contains the *_sat.jpg and *_mask.png pairs.")
print("For convenience, copying can take additional disk space; the project intentionally leaves the original cache untouched.")
