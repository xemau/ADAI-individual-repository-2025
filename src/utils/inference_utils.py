import torch
import torchvision.transforms.functional as TF
import numpy as np


@torch.no_grad()
def predict_with_tta(model, xb, device):
    xb = xb.to(device)
    out = model(xb)
    prob = torch.softmax(out, dim=1)
    xb_flip = TF.hflip(xb)
    out_flip = model(xb_flip)
    prob_flip = torch.softmax(out_flip, dim=1)
    prob_avg = (prob + prob_flip) / 2.0
    return prob_avg

@torch.no_grad()
def collect_outputs(model, loader, device, use_tta=True):
    ys, preds, probs = [], [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            xb, yb = batch
            if use_tta:
                pr = predict_with_tta(model, xb, device).cpu()
            else:
                out = model(xb.to(device))
                pr = torch.softmax(out, dim=1).cpu()
            pb = pr[:, 1]
            preds.extend(pr.argmax(dim=1).tolist())
            probs.extend(pb.tolist())
            ys.extend(yb.tolist())
        else:
            xb = batch
            if use_tta:
                pr = predict_with_tta(model, xb, device).cpu()
            else:
                out = model(xb.to(device))
                pr = torch.softmax(out, dim=1).cpu()
            pb = pr[:, 1]
            preds.extend(pr.argmax(dim=1).tolist())
            probs.extend(pb.tolist())
    return ys if len(ys) else None, preds, probs

@torch.no_grad()
def predict_with_tta_mc(model, xb, device):
    xb = xb.to(device)
    out = model(xb)
    prob = torch.softmax(out, dim=1)
    xb_flip = TF.hflip(xb)
    out_flip = model(xb_flip)
    prob_flip = torch.softmax(out_flip, dim=1)
    prob_avg = (prob + prob_flip) / 2.0
    return prob_avg

@torch.no_grad()
def collect_outputs_mc(model, loader, device, use_tta=True):
    ys, preds, probs = [], [], []
    for xb, yb in loader:
        if use_tta:
            pr = predict_with_tta_mc(model, xb, device).cpu()
        else:
            out = model(xb.to(device))
            pr = torch.softmax(out, dim=1).cpu()
        preds.extend(pr.argmax(dim=1).tolist())
        probs.append(pr.numpy())
        ys.extend(yb.tolist())
    probs = np.concatenate(probs, axis=0)
    return ys, preds, probs