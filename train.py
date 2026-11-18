import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from satseg.data import make_loaders
from satseg.models import build_model
from satseg.losses import segmentation_loss
from satseg.metrics import batch_metrics, mean_metrics

def main():
    p = argparse.ArgumentParser(description="Train satellite segmentation models")
    p.add_argument("--task", choices=["roads", "buildings"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", choices=["unet", "attention_unet"], default="attention_unet")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-deep-supervision", action="store_true")
    args = p.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, total = make_loaders(args.data_root, args.image_size, args.batch_size, workers=args.workers)
    model = build_model(args.model, not args.no_deep_supervision).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    outdir = Path("artifacts") / args.task / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    best_iou = -1
    history = []
    print(f"Device: {device} | Paired samples: {total} | Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses=[]
        for x,y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} train"):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(x); loss=segmentation_loss(outputs,y)
            loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        model.eval(); vals=[]; val_losses=[]
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device)
                out=model(x)
                if isinstance(out,dict): out=out["out"]
                val_losses.append(segmentation_loss(out,y).item())
                vals.append(batch_metrics(out,y))
        metrics=mean_metrics(vals)
        row={"epoch":epoch,"train_loss":float(np.mean(train_losses)),"val_loss":float(np.mean(val_losses)),**metrics}
        history.append(row); print(json.dumps(row, indent=2))
        torch.save({"model_state":model.state_dict(),"model":args.model,"task":args.task,"epoch":epoch,"metrics":metrics},outdir/"last.pt")
        if metrics.get("iou",0)>best_iou:
            best_iou=metrics["iou"]
            torch.save({"model_state":model.state_dict(),"model":args.model,"task":args.task,"epoch":epoch,"metrics":metrics},outdir/"best.pt")
        (outdir/"history.json").write_text(json.dumps(history,indent=2))
    print(f"Training complete. Best validation IoU: {best_iou:.4f}. Checkpoint: {outdir/'best.pt'}")

if __name__=="__main__": main()
