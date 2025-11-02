# ================================================================
# main_clip_fusion_v7.py
# CLIP (ViT-B/16) + Deep Fusion (A-32) with CUDA, AMP, caching, bin SMAPE
# ================================================================

import os
import re
import io
import math
import argparse
import hashlib
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
import requests
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------- Utils --------------------------------------

PRICE_BINS = [0, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def file_hash(path_list: List[str]) -> str:
    h = hashlib.sha1()
    for p in path_list:
        h.update(str(p).encode())
    return h.hexdigest()[:12]

def extract_value_and_unit(text):
    text = str(text).lower()
    value_match = re.search(r"value:\s*([\d.,]+)", text)
    unit_match = re.search(r"unit:\s*([a-zA-Z]+)", text)
    try:
        value = float(value_match.group(1).replace(',', '.')) if value_match else np.nan
    except Exception:
        value = np.nan
    unit = unit_match.group(1).lower() if unit_match else None

    multiplier = 1.0
    if unit in ['kg', 'kilogram', 'kilograms']:
        multiplier = 1000.0
    elif unit in ['l', 'litre', 'liter']:
        multiplier = 1000.0
    elif unit in ['ml', 'ounce', 'oz', 'ounces', 'fl', 'fl oz', 'fl_oz', 'count', 'pack']:
        multiplier = 1.0
    return (value * multiplier) if (not np.isnan(value)) else 0.0, unit

def smape_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    numer = np.abs(y_pred - y_true)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.zeros_like(numer)
        nz = denom != 0
        ratio[nz] = numer[nz] / denom[nz]
        ratio[~nz] = 0.0
    return float(np.mean(ratio))

def safe_expm1_clip(x, min_val=-20.0, max_val=20.0):
    x_cl = np.clip(x, min_val, max_val)
    return np.expm1(x_cl)

def download_image(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return None

# ------------------------ Embedding (stream + cache) -------------------------

def probe_image_dim(clip_model) -> int:
    dummy = Image.new("RGB", (224, 224), (0, 0, 0))
    try:
        d = clip_model.encode([dummy], convert_to_numpy=True).shape[1]
        return int(d)
    except Exception:
        # Fallback to text dimension; SentenceTransformers keeps them equal for CLIP
        return int(clip_model.get_sentence_embedding_dimension())

def compute_clip_embeddings_stream(clip_model, texts, image_links, batch_size=64) -> Tuple[np.ndarray, np.ndarray]:
    # Text
    text_embs = clip_model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    # Image (low-RAM streaming)
    n = len(image_links)
    dim = probe_image_dim(clip_model)
    print(f"✅ Image embedding dim: {dim}")
    img_embs = np.zeros((n, dim), dtype=np.float32)
    dummy = Image.new("RGB", (224, 224), (0, 0, 0))

    imgs, idxs = [], []

    def flush():
        nonlocal imgs, idxs
        if not imgs:
            return
        try:
            emb = clip_model.encode(imgs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print("⚠️ Image encode failed; using zeros for this batch:", e)
            emb = np.zeros((len(imgs), dim), dtype=np.float32)
        img_embs[idxs] = emb.astype(np.float32)
        imgs, idxs = [], []

    for i, url in enumerate(tqdm(image_links, desc="Images (stream)")):
        img = download_image(url)
        if img is None:
            img = dummy
        imgs.append(img)
        idxs.append(i)
        if len(imgs) >= batch_size:
            flush()
    flush()
    return text_embs.astype(np.float32), img_embs.astype(np.float32)

def compute_clip_embeddings_cached(split_name: str,
                                    clip_model,
                                    texts: List[str],
                                    image_links: List[str],
                                    cache_dir: str,
                                    clip_model_name: str,
                                    batch_size: int = 64):
    ensure_dir(cache_dir)
    key = f"{split_name}__{clip_model_name}__n{len(texts)}__{file_hash(texts[:10] + image_links[:10])}"
    text_cache = os.path.join(cache_dir, f"{key}_text.npy")
    img_cache  = os.path.join(cache_dir, f"{key}_img.npy")

    have_text = os.path.exists(text_cache)
    have_img  = os.path.exists(img_cache)

    text_embs = None
    img_embs = None

    if have_text and have_img:
        print(f"[{split_name}] Using cached TEXT+IMAGE embeddings from {cache_dir}")
        text_embs = np.load(text_cache, mmap_mode="r").astype(np.float32)
        img_embs  = np.load(img_cache, mmap_mode="r").astype(np.float32)
        return text_embs, img_embs

    if not have_text:
        print(f"[{split_name}] Encoding TEXT with CLIP (no valid cache)…")
    if not have_img:
        print(f"[{split_name}] Encoding IMAGES with CLIP in low-memory batches (no valid cache)…")

    # If one is missing, recompute both in a single stream call to keep consistency
    t_embs, i_embs = compute_clip_embeddings_stream(clip_model, texts, image_links, batch_size=batch_size)
    np.save(text_cache, t_embs)
    np.save(img_cache,  i_embs)
    print(f"[{split_name}] Saved caches -> {text_cache} & {img_cache}")
    return t_embs, i_embs

# ------------------------------- Dataset ------------------------------------

class PriceDataset(Dataset):
    def __init__(self, df, text_emb, img_emb, scaler, is_train=True):
        assert len(df) == len(text_emb) == len(img_emb), "Embeddings length must match dataframe rows"
        tab = df[['value', 'has_image']].fillna(0).values.astype(np.float32)
        self.tab = scaler.transform(tab).astype(np.float32)
        self.text_emb = text_emb.astype(np.float32)
        self.img_emb = img_emb.astype(np.float32)
        self.is_train = is_train
        self.y = (np.log1p(df['price'].values).astype(np.float32)) if is_train else None

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):
        tab = torch.from_numpy(self.tab[idx])
        text = torch.from_numpy(self.text_emb[idx])
        img = torch.from_numpy(self.img_emb[idx])
        if self.is_train:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return text, img, tab, y
        return text, img, tab

# ------------------------------- Model (A-32) --------------------------------

class FusionBlock(nn.Module):
    def __init__(self, dim, hidden_mult=2, drop=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(drop),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff2 = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.ff1(self.norm1(x))
        x = x + self.ff2(self.norm2(x))
        return x

class DeepFusionRegressor(nn.Module):
    def __init__(self, embed_dim=512, tab_dim=2, width=768, depth=3, drop=0.2):
        super().__init__()
        # projectors
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, width),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(embed_dim, width),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, width // 2),
            nn.GELU(),
            nn.Dropout(drop),
        )
        fused_dim = width * 2 + (width // 2)

        self.blocks = nn.Sequential(*[FusionBlock(fused_dim, hidden_mult=2, drop=drop) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, width),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(width, 1),
        )

    def forward(self, text, img, tab):
        t = self.text_proj(text)
        i = self.img_proj(img)
        h = self.tab_proj(tab)
        x = torch.cat([t, i, h], dim=1)
        x = self.blocks(x)
        out = self.head(x)
        return out.squeeze(1)

# ------------------------------- Train / Eval --------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip=5.0, amp=True):
    model.train()
    total = 0.0
    for batch in loader:
        text, img, tab, y = batch
        text = text.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            preds = model(text, img, tab)
            loss = criterion(preds, y)

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, clamp_min=-20.0, clamp_max=20.0, amp=True):
    model.eval()
    total = 0.0
    preds_all, trues_all = [], []
    for batch in loader:
        text, img, tab, y = batch
        text = text.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(text, img, tab)
            loss = criterion(out, y)
        total += loss.item()
        preds_all.append(out.detach().cpu().numpy())
        trues_all.append(y.detach().cpu().numpy())
    if not preds_all:
        return float('nan'), float('nan')
    preds = np.concatenate(preds_all)
    trues = np.concatenate(trues_all)

    preds_prices = safe_expm1_clip(np.clip(preds, clamp_min, clamp_max), clamp_min, clamp_max)
    trues_prices = safe_expm1_clip(np.clip(trues, clamp_min, clamp_max), clamp_min, clamp_max)
    return total / max(1, len(loader)), smape_safe(trues_prices, preds_prices)

def smape_per_bin(y_true_price: np.ndarray, y_pred_price: np.ndarray, bins=PRICE_BINS):
    y_true = np.asarray(y_true_price, dtype=float)
    y_pred = np.asarray(y_pred_price, dtype=float)
    out = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            out.append((f"[{lo},{hi})", np.nan, 0))
        else:
            out.append((f"[{lo},{hi})", smape_safe(y_true[mask], y_pred[mask]), int(mask.sum())))
    return out

# --------------------------------- Main -------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=str, default='dataset/train.csv')
    p.add_argument('--test', type=str, default='dataset/test.csv')
    p.add_argument('--out', type=str, default='dataset/sample_test_clip_v7.csv')
    p.add_argument('--cache-dir', type=str, default='cache_clip_v7')
    p.add_argument('--artifacts', type=str, default='artifacts')
    p.add_argument('--clip-model', type=str, default='clip-ViT-B-16')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--amp', type=lambda s: s.lower() in ['1','true','yes'], default=True)
    args = p.parse_args()

    ensure_dir(args.artifacts)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 Running CLIP + Tabular Fusion (v7)")
    print(f"Using device: {device}")
    print(f"CLIP model: {args.clip-model if hasattr(args, 'clip-model') else args.clip_model}")

    # Data
    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)

    train_df['value'], train_df['unit'] = zip(*train_df['catalog_content'].map(extract_value_and_unit))
    test_df['value'],  test_df['unit']  = zip(*test_df['catalog_content'].map(extract_value_and_unit))
    train_df['has_image'] = (~train_df['image_link'].isna()).astype(int)
    test_df['has_image']  = (~test_df['image_link'].isna()).astype(int)

    # CLIP model
    print("Loading CLIP model (SentenceTransformers)…")
    clip_model = SentenceTransformer(args.clip_model, device=device)

    # Caching embeddings
    print("🔁 Building / loading cached embeddings…")
    tr_text, tr_img = compute_clip_embeddings_cached(
        split_name="train",
        clip_model=clip_model,
        texts=train_df['catalog_content'].astype(str).tolist(),
        image_links=train_df['image_link'].astype(str).tolist(),
        cache_dir=args.cache_dir,
        clip_model_name=args.clip_model,
        batch_size=64,
    )
    te_text, te_img = compute_clip_embeddings_cached(
        split_name="test",
        clip_model=clip_model,
        texts=test_df['catalog_content'].astype(str).tolist(),
        image_links=test_df['image_link'].astype(str).tolist(),
        cache_dir=args.cache_dir,
        clip_model_name=args.clip_model,
        batch_size=64,
    )

    # Scale tabular
    scaler = StandardScaler()
    all_tab = np.vstack([
        train_df[['value', 'has_image']].fillna(0).values,
        test_df[['value', 'has_image']].fillna(0).values
    ]).astype(np.float32)
    scaler.fit(all_tab)

    # Split
    idx = np.arange(len(train_df))
    tr_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42, shuffle=True)
    X_tr = train_df.iloc[tr_idx].reset_index(drop=True)
    X_va = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds = PriceDataset(X_tr, tr_text[tr_idx], tr_img[tr_idx], scaler, is_train=True)
    val_ds   = PriceDataset(X_va, tr_text[val_idx], tr_img[val_idx], scaler, is_train=True)
    test_ds  = PriceDataset(test_df.reset_index(drop=True), te_text, te_img, scaler, is_train=False)

    pin = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin)

    embed_dim = tr_text.shape[1]
    print(f"Embedding dim: {embed_dim}")

    # Model
    model = DeepFusionRegressor(embed_dim=embed_dim, tab_dim=2,
                                width=args.width, depth=args.depth, drop=args.dropout).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda' and args.amp))

    best_smape = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, amp=args.amp)
        va_loss, va_smape = eval_epoch(model, val_loader, criterion, device, amp=args.amp)
        scheduler.step(epoch)

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_smape": va_smape})
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | SMAPE: {va_smape:.6f}")

        # save checkpoints
        torch.save(model.state_dict(), os.path.join(args.artifacts, f"epoch_{epoch:03d}.pth"))
        if np.isfinite(va_smape) and va_smape < best_smape:
            best_smape = va_smape
            torch.save(model.state_dict(), os.path.join(args.artifacts, "best.pth"))

    # Save training curves and history
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(args.artifacts, "metrics_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    try:
        plt.figure(figsize=(8,4))
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
        plt.twinx()
        plt.plot(hist_df["epoch"], hist_df["val_smape"], label="val_smape", linestyle="--")
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        prog_png = os.path.join(args.artifacts, "progress.png")
        plt.tight_layout()
        plt.savefig(prog_png, dpi=150)
        plt.close()
    except Exception:
        pass

    # Load best model
    best_path = os.path.join(args.artifacts, "best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"✅ Loaded best model: {best_path}")

    # SMAPE by price bin on validation
    model.eval()
    va_preds_log = []
    va_trues_log = []
    with torch.no_grad():
        for text, img, tab, y in val_loader:
            text = text.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            tab = tab.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(text, img, tab)
            va_preds_log.append(out.cpu().numpy())
            va_trues_log.append(y.numpy())
    va_preds_log = np.concatenate(va_preds_log)
    va_trues_log = np.concatenate(va_trues_log)
    va_preds_price = safe_expm1_clip(np.clip(va_preds_log, -20, 20), -20, 20)
    va_trues_price = safe_expm1_clip(np.clip(va_trues_log, -20, 20), -20, 20)

    bin_rows = smape_per_bin(va_trues_price, va_preds_price, bins=PRICE_BINS)
    bin_df = pd.DataFrame(bin_rows, columns=["bin", "smape", "count"])
    bin_csv = os.path.join(args.artifacts, "val_bin_smape.csv")
    bin_df.to_csv(bin_csv, index=False)

    # Predict on test
    preds_log = []
    with torch.no_grad():
        for text, img, tab in test_loader:
            text = text.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            tab = tab.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(text, img, tab)
            preds_log.append(out.cpu().numpy())
    preds_log = np.concatenate(preds_log)
    preds_price = safe_expm1_clip(np.clip(preds_log, -20.0, 20.0), -20.0, 20.0)
    preds_price[preds_price <= 0] = 1e-3

    out_df = pd.DataFrame({'sample_id': test_df['sample_id'].values, 'price': preds_price})
    out_df.to_csv(args.out, index=False)

    print(f"📉 Best validation SMAPE: {best_smape:.6f}")
    print(f"💾 Saved predictions to {args.out}")
    print(f"🖼️ Progress plot: {os.path.join(args.artifacts, 'progress.png')}")
    print(f"📑 Metrics CSV:   {hist_csv}")
    print(f"📑 Bin SMAPE CSV: {bin_csv}")

if __name__ == "__main__":
    main()
