import argparse
from pathlib import Path
import cv2
from satseg.classical import classical_extract

p=argparse.ArgumentParser()
p.add_argument("--image",required=True)
p.add_argument("--task",choices=["roads","buildings"],default="roads")
p.add_argument("--output",default="classical_mask.png")
a=p.parse_args()
img=cv2.cvtColor(cv2.imread(a.image),cv2.COLOR_BGR2RGB)
mask=classical_extract(img,a.task)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
cv2.imwrite(a.output,mask)
print(f"Saved classical baseline mask to {a.output}")
