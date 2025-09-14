#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, glob, argparse
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.backends.cudnn as cudnn

def build_amp():
    try:
        from torch.amp import autocast as ac_new, GradScaler as GS_new
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            scaler = GS_new(device_type)
        except TypeError:
            scaler = GS_new(enabled=torch.cuda.is_available())
        def ctx(): return ac_new(device_type)
        return ctx, scaler
    except Exception:
        from torch.cuda.amp import autocast as ac_old, GradScaler as GS_old
        scaler = GS_old(enabled=torch.cuda.is_available())
        def ctx(): return ac_old(enabled=torch.cuda.is_available())
        return ctx, scaler

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_img_root(base_dir: str) -> str:
    cands = [
        os.path.join(base_dir, "images_leadI"),
        os.path.join(base_dir, "images_leadl"),
    ] + glob.glob(os.path.join(base_dir, "images_lead*"))
    for c in cands:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"No se encontró carpeta de imágenes en {base_dir} (images_lead*).")

def load_labels_and_fix_paths(labels_csv: str, img_root: str) -> pd.DataFrame:
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"No existe {labels_csv}")
    df = pd.read_csv(labels_csv)
    for col in ("label", "ecg_id"):
        if col not in df.columns:
            raise ValueError("labels.csv debe contener 'label' y 'ecg_id'.")
    if "filepath" not in df.columns:
        df["filepath"] = ""
    def ok(p): 
        try: return os.path.exists(p)
        except: return False
    bad = ~df["filepath"].apply(ok)
    if bad.any():
        def rebuild(row):
            ecg = str(row["ecg_id"])
            if ecg.endswith(".0"): ecg = ecg[:-2]
            return os.path.join(img_root, str(row["label"]), f"{ecg}.png")
        df.loc[bad, "filepath"] = df[bad].apply(rebuild, axis=1)
        bad2 = ~df["filepath"].apply(ok)
        if bad2.any():
            sample = df[bad2][["ecg_id","label","filepath"]].head(10)
            raise FileNotFoundError(f"Rutas inexistentes ({bad2.sum()}). Ejemplos:\n{sample.to_string(index=False)}")
    return df

def stratified_split(df: pd.DataFrame, split):
    parts = []
    for label, grp in df.groupby("label", sort=False):
        grp = grp.sample(frac=1.0, random_state=split.get("seed", 42))
        n = len(grp)
        if n < 5:
            n_train, n_val = n, 0
        else:
            n_train = max(1, int(n * split["train"]))
            n_val   = int(n * split["val"])
            if n >= 10 and n_val == 0: n_val = 1
            n_train = min(n_train, n - n_val)
            if n_train < 1: n_train = 1
            if n_train + n_val > n: n_val = max(0, n - n_train)
        g_train = grp.iloc[:n_train]
        g_val   = grp.iloc[n_train:n_train+n_val]
        g_test  = grp.iloc[n_train+n_val:]
        parts += [("train", g_train), ("val", g_val), ("test", g_test)]
    df_train = pd.concat([g for p,g in parts if p=="train"]).reset_index(drop=True)
    df_val   = pd.concat([g for p,g in parts if p=="val"]).reset_index(drop=True)
    df_test  = pd.concat([g for p,g in parts if p=="test"]).reset_index(drop=True)
    return df_train, df_val, df_test

def compute_class_weights(counts: pd.Series) -> torch.Tensor:
    freq = counts.astype(float).copy()
    if (freq > 0).any():
        min_nz = freq[freq > 0].min()
        freq[freq == 0] = min_nz
    else:
        freq[:] = 1.0
    N, C = freq.sum(), len(freq)
    weights = N / (C * freq)
    return torch.tensor(weights.values, dtype=torch.float32)

def make_weighted_sampler(labels_idx: List[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(np.array(labels_idx), minlength=num_classes).astype(float)
    inv = 1.0 / (counts + 1e-9)
    w = inv[np.array(labels_idx)]
    return WeightedRandomSampler(weights=torch.from_numpy(w),
                                 num_samples=len(w),
                                 replacement=True)

class ECGImageCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout2(x)
        return self.classifier(x)

class CSVImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int], img_size: Tuple[int, int]):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["filepath"])
        img = self.tf(img)
        y = self.label2idx[r["label"]]
        return img, y

def train_one_epoch(model, loader, optimizer, criterion, device, autocast_ctx, scaler):
    model.train()
    loss_sum = acc_sum = 0.0
    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(imgs)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * imgs.size(0)
        acc_sum  += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, acc_sum / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = acc_sum = 0.0
    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        logits = model(imgs)
        loss = criterion(logits, y)
        loss_sum += loss.item() * imgs.size(0)
        acc_sum  += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, acc_sum / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))   # /opt/ml/input/data/training
    parser.add_argument("--model_dir",  type=str, default=os.environ.get("SM_MODEL_DIR") or "./output_model")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR") or "./output_data")
    parser.add_argument("--local_output", type=str, default="./output")
    parser.add_argument("--img_h", type=int, default=450)
    parser.add_argument("--img_w", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sampler", action="store_true")
    args = parser.parse_args()

    cudnn.benchmark = True
    set_seed(args.seed)

    if args.data_dir:   # SageMaker
        BASE = args.data_dir
        MODEL_DIR = args.model_dir
        OUT_DIR = args.output_dir
    else:               # Local
        BASE = args.local_output   # espera labels.csv + images_leadI/
        MODEL_DIR = os.path.join(args.local_output, "runs_ecg")
        OUT_DIR = os.path.join(args.local_output, "runs_ecg")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    labels_csv = os.path.join(BASE, "labels.csv")
    img_root   = resolve_img_root(BASE)

    print("DATA BASE   =", BASE)
    print("IMG_ROOT    =", img_root)
    print("MODEL_DIR   =", MODEL_DIR)
    print("OUT_DIR     =", OUT_DIR)

    df = load_labels_and_fix_paths(labels_csv, img_root)
    if df.empty:
        raise RuntimeError("labels.csv vacío.")

    classes = sorted(df["label"].unique())
    label2idx = {c:i for i,c in enumerate(classes)}
    num_classes = len(classes)

    split = {"train":0.8, "val":0.1, "test":0.1, "seed":args.seed}
    df_train, df_val, df_test = stratified_split(df, split)

    ds_train = CSVImageDataset(df_train, label2idx, (args.img_h, args.img_w))
    ds_val   = CSVImageDataset(df_val,   label2idx, (args.img_h, args.img_w))
    ds_test  = CSVImageDataset(df_test,  label2idx, (args.img_h, args.img_w))

    train_labels_idx = [label2idx[l] for l in df_train["label"].tolist()]
    if args.use_sampler:
        sampler = make_weighted_sampler(train_labels_idx, num_classes)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    else:
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    dl_val  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    counts = df_train["label"].value_counts().reindex(classes, fill_value=0)
    class_weights = compute_class_weights(counts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGImageCNN(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    autocast_ctx, scaler = build_amp()

    best_val = 0.0
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, criterion, device, autocast_ctx, scaler)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)
        print(f"[{epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}", flush=True)

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes,
                "label2idx": label2idx,
                "val_acc": best_val,
                "img_size": (args.img_h, args.img_w),
            }, ckpt_path)
            print(f"✔ Guardado mejor modelo en {ckpt_path} (val_acc={best_val:.4f})", flush=True)

    te_loss, te_acc = evaluate(model, dl_test, criterion, device)
    print(f"TEST: loss={te_loss:.4f} acc={te_acc:.4f}", flush=True)

    # Artefactos para el endpoint
    with open(os.path.join(MODEL_DIR, "label2idx.json"), "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    meta = {"img_size": [args.img_h, args.img_w], "classes": classes, "val_acc": float(best_val), "test_acc": float(te_acc)}
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"val_acc": float(best_val), "test_acc": float(te_acc)}, f)

if __name__ == "__main__":
    main()
