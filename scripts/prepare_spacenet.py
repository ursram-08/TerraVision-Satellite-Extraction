"""Rasterize SpaceNet GeoJSON building polygons to binary masks using matching GeoTIFF georeferencing."""
import argparse
from pathlib import Path
import json
import rasterio
from rasterio.features import rasterize

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True,help="Directory containing GeoTIFF imagery")
    p.add_argument("--labels",required=True,help="Directory containing SpaceNet GeoJSON labels")
    p.add_argument("--output",default="data/spacenet2/paired")
    p.add_argument("--image-pattern",default="*.tif")
    a=p.parse_args()
    images=Path(a.images); labels=Path(a.labels); out=Path(a.output)
    (out/"images").mkdir(parents=True,exist_ok=True); (out/"masks").mkdir(parents=True,exist_ok=True)
    geojsons=list(labels.rglob("*.geojson"))
    done=0
    for img in images.rglob(a.image_pattern):
        # Common SpaceNet label naming convention: <image-stem>.geojson or same stem in a labels folder.
        candidates=[labels/(img.stem+".geojson"), *labels.rglob(img.stem+".geojson")]
        gj=next((x for x in candidates if x.exists()),None)
        if gj is None: continue
        with rasterio.open(img) as src:
            transform, shape, crs = src.transform, (src.height,src.width), src.crs
            data=json.loads(gj.read_text(encoding="utf-8"))
            features=[]
            for f in data.get("features",[]):
                geom=f.get("geometry")
                if geom: features.append((geom,1))
            mask=rasterize(features,out_shape=shape,transform=transform,fill=0,dtype="uint8")
            # Save RGB preview-compatible image; preserve image bands by writing first 3 bands to PNG.
            arr=src.read()
            if arr.shape[0] >= 3:
                rgb=arr[:3].transpose(1,2,0)
                import numpy as np
                rgb=np.nan_to_num(rgb).astype("float32")
                lo,hi=np.percentile(rgb,[2,98])
                rgb=((rgb-lo)/max(hi-lo,1e-6)*255).clip(0,255).astype("uint8")
            else:
                import numpy as np
                rgb=np.repeat(arr[0][...,None],3,axis=2).astype("uint8")
        from PIL import Image
        Image.fromarray(rgb).save(out/"images"/(img.stem+".png"))
        Image.fromarray(mask*255).save(out/"masks"/(img.stem+".png"))
        done+=1
    print(f"Prepared {done} image/mask pairs at {out}. GeoJSON files discovered: {len(geojsons)}")
    if done==0:
        print("No pairs found. SpaceNet archive layouts and label filenames vary; inspect your AOI and adjust image/label paths or naming rules.")

if __name__=="__main__": main()
