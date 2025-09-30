import os
from typing import Optional, Dict, Tuple, List

import kagglehub
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value, Image as HFImage


def _download_bcn20000() -> str:
    return kagglehub.dataset_download("pasutchien/bcn20000")


def _dataset_roots(base: Optional[str] = None) -> Tuple[str, str, str, str, str]:
    root = base or _download_bcn20000()
    train_imgs = os.path.join(root, "BCN_20k_train", "bcn_20k_train")
    test_imgs = os.path.join(root, "BCN_20k_test", "bcn_20k_test")
    train_csv = os.path.join(root, "bcn_20k_train.csv")
    test_csv = os.path.join(root, "bcn_20k_test.csv")
    if not os.path.isdir(train_imgs) or not os.path.isdir(test_imgs):
        raise FileNotFoundError("Missing expected image folders")
    if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
        raise FileNotFoundError("Missing expected CSV files")
    return root, train_imgs, test_imgs, train_csv, test_csv


def _resolve_path(images_root: str, name: str) -> Optional[str]:
    name = str(name)
    base, ext = os.path.splitext(name)
    candidates: List[str] = []
    if ext:
        candidates.append(name)
    else:
        for e in (".jpg", ".jpeg", ".png", ".bmp"):
            candidates.append(base + e)
    for cand in candidates:
        p = os.path.join(images_root, cand)
        if os.path.exists(p):
            return p
    p_rel = os.path.join(images_root, name)
    if os.path.exists(p_rel):
        return p_rel
    return None


def _build_dataset(csv_path: str, images_root: str,
                   filename_column: str, label_column: Optional[str],
                   label_mapping: Optional[Dict[str, str]] = None) -> Dataset:
    df = pd.read_csv(csv_path)

    # always keep filename; only keep label if provided and present
    keep_cols = [filename_column]
    has_labels = label_column is not None and label_column in df.columns
    if has_labels:
        keep_cols.append(label_column)

    df = df[keep_cols].copy()

    # optional mapping for labeled splits
    if has_labels and label_mapping:
        df[label_column] = df[label_column].astype(str).map(label_mapping)

    # drop rows with missing label (if labeled) and always resolve image paths
    if has_labels:
        df = df.dropna(subset=[label_column])

    def _resolve(name: str) -> Optional[str]:
        name = str(name)
        base, ext = os.path.splitext(name)
        candidates = [name] if ext else [base + e for e in (".jpg", ".jpeg", ".png", ".bmp")]
        for cand in candidates:
            p = os.path.join(images_root, cand)
            if os.path.exists(p):
                return p
        p_rel = os.path.join(images_root, name)
        return p_rel if os.path.exists(p_rel) else None

    df["image_path"] = df[filename_column].apply(_resolve)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # build records and cast features
    if has_labels:
        records = [
            {"image": row["image_path"], "label": str(row[label_column]), "image_path": row["image_path"]}
            for _, row in df.iterrows()
        ]
        ds = Dataset.from_list(records)
        label_names = sorted(set(df[label_column].astype(str).tolist()))
        features = Features({
            "image": HFImage(),
            "label": ClassLabel(names=label_names),
            "image_path": Value("string"),
        })
        return ds.cast(features)
    else:
        records = [
            {"image": row["image_path"], "image_path": row["image_path"]}
            for _, row in df.iterrows()
        ]
        ds = Dataset.from_list(records)
        features = Features({
            "image": HFImage(),
            "image_path": Value("string"),
        })
        return ds.cast(features)

def load_bcn20000(
    split: str = "train",
    base_dir: Optional[str] = None,
    filename_column: str = "bcn_filename",
    label_column: Optional[str] = "diagnosis",
    label_mapping: Optional[Dict[str, str]] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dataset:
    root, train_imgs, test_imgs, train_csv, test_csv = _dataset_roots(base_dir)
    split = split.lower()
    if split == "val":
        split = "validation"

    if split == "train":
        return _build_dataset(train_csv, train_imgs, filename_column, label_column, label_mapping)

    if split == "test":
        return _build_dataset(test_csv, test_imgs, filename_column, None, None)

    if split == "validation":
        train_ds = _build_dataset(train_csv, train_imgs, filename_column, label_column, label_mapping)
        label_names = train_ds.features["label"].names
        by_label: Dict[int, List[int]] = {i: [] for i in range(len(label_names))}
        for idx in range(len(train_ds)):
            ex = train_ds[idx]
            lbl = ex["label"]
            if isinstance(lbl, str):
                lbl = label_names.index(lbl)
            by_label[int(lbl)].append(idx)
        import random
        rng = random.Random(seed)
        val_indices: List[int] = []
        for _, idxs in by_label.items():
            if not idxs:
                continue
            n_val = max(1, int(len(idxs) * val_ratio))
            rng.shuffle(idxs)
            val_indices.extend(idxs[:n_val])
        return train_ds.select(sorted(val_indices))

    raise ValueError("split must be one of {'train','validation','val','test'}")