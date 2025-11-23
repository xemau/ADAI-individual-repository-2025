from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src import load_bcn20000, get_transforms

class TorchImageDataset(Dataset):
    def __init__(self, hf_ds, transform, has_labels=True):
        self.ds = hf_ds
        self.tf = transform
        self.has_labels = has_labels
        self.label_feature = hf_ds.features["label"] if has_labels else None
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        x = self.tf(img)
        if self.has_labels:
            y = ex["label"]
            if isinstance(y, str):
                y = self.label_feature.str2int(y)
            else:
                y = int(y)
            return x, y
        return x

def get_binary_mapping():
    malignant = {"MEL","SCC","BCC"}
    all_labels = ["MEL","SCC","NV","BCC","BKL","AK","DF","VASC"]
    return {lbl: ("malignant" if lbl in malignant else "benign") for lbl in all_labels}

def make_loader(split, batch_size=64, shuffle=False, labeled=True):
    label_mapping = get_binary_mapping() if labeled else None
    label_column = "diagnosis" if labeled else None
    hf = load_bcn20000(split=split, filename_column="bcn_filename", label_column=label_column, label_mapping=label_mapping)
    tf = get_transforms(train=(split=="train"))
    ds = TorchImageDataset(hf, tf, has_labels=labeled)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, hf.features["label"].names if labeled else None

def make_train_val_loaders(batch_size=64):
    train_loader, label_names = make_loader("train", batch_size=batch_size, shuffle=True, labeled=True)
    val_loader, _ = make_loader("validation", batch_size=batch_size, shuffle=False, labeled=True)
    return train_loader, val_loader, label_names

def make_loader_multiclass(split, batch_size=64, shuffle=False):
    hf = load_bcn20000(split=split, filename_column="bcn_filename", label_column="diagnosis", label_mapping=None)
    tf = get_transforms(train=(split=="train"))
    ds = TorchImageDataset(hf, tf, has_labels=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, hf.features["label"].names

def make_train_val_loaders_multiclass(batch_size=64):
    train_loader, label_names = make_loader_multiclass("train", batch_size=batch_size, shuffle=True)
    val_loader, _ = make_loader_multiclass("validation", batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, label_names