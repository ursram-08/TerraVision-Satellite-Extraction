import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
from satseg.data import make_loaders
from satseg.models import build_model
from satseg.metrics import batch_metrics, mean_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["roads","buildings"],required=True)
    p.add_argument("--data-root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--model",choices=["unet","attention_unet"],default="attention_unet")
    p.add_argument("--image-size",type=int,default=512)
    args=p.parse_args()
    _, loader, n=make_loaders(args.data_root,args.image_size,batch_size=4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(args.model,False).to(device)
    ckpt=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ckpt["model_state"]); model.eval()
    rows=[]
    with torch.no_grad():
        for x,y in tqdm(loader,desc="Evaluating"):
            out=model(x.to(device)); rows.append(batch_metrics(out,y.to(device)))
    result={"task":args.task,"samples_total":n,"metrics":mean_metrics(rows)}
    print(json.dumps(result,indent=2))
    Path(args.checkpoint).with_name("evaluation.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__": main()
