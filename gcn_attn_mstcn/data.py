import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class StandardScaler:    
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def load_metr_la(csv_path: str, interpolate_missing: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if interpolate_missing:
        df = df.replace(0, np.nan).interpolate(method="linear", limit_direction="both")
        df = df.bfill().ffill()
    return df


def generate_seq2seq_io_data(df, x_offsets, y_offsets,
                             add_time_in_day=True, add_day_in_week=True):
    num_samples, num_nodes = df.shape
    feats = [np.expand_dims(df.values, axis=-1)]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        feats.append(np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)))
    if add_day_in_week:
        dow = df.index.dayofweek.values / 6.0
        feats.append(np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0)))

    data = np.concatenate(feats, axis=-1).astype(np.float32)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    x = np.stack([data[t + x_offsets] for t in range(min_t, max_t)])
    y = np.stack([data[t + y_offsets][..., 0] for t in range(min_t, max_t)])
    return x, y


def load_adjacency(pkl_path: str):
    with open(pkl_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj = pickle.load(f, encoding="latin1")
    return sensor_ids, sensor_id_to_ind, np.asarray(adj, dtype=np.float32)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    n = adj.size(0)
    a_hat = adj + torch.eye(n, device=adj.device)
    deg_inv_sqrt = a_hat.sum(dim=1).pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    d = torch.diag(deg_inv_sqrt)
    return d @ a_hat @ d


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    scaler: StandardScaler
    num_nodes: int
    num_features: int


def build_dataloaders(cfg: dict) -> DataBundle:
    dcfg = cfg["data"]
    df = load_metr_la(dcfg["csv_path"], dcfg.get("interpolate_missing", False))

    x_offsets = np.arange(-(dcfg["input_len"] - 1), 1)
    y_offsets = np.arange(1, dcfg["output_len"] + 1)
    x, y_raw = generate_seq2seq_io_data(df, x_offsets, y_offsets)

    n = x.shape[0]
    n_train = round(n * dcfg["train_ratio"])
    n_test = round(n * dcfg["test_ratio"])
    n_val = n - n_train - n_test

    splits = {
        "train": (x[:n_train], y_raw[:n_train]),
        "val": (x[n_train:n_train + n_val], y_raw[n_train:n_train + n_val]),
        "test": (x[-n_test:], y_raw[-n_test:]),
    }

    # one scaler, fitted on the training speed channel only
    train_speed = splits["train"][0][..., 0]
    scaler = StandardScaler(train_speed.mean(), train_speed.std())

    bs = cfg["train"]["batch_size"]
    loaders = {}
    for name, (xs, ys_raw) in splits.items():
        xs = xs.copy()
        xs[..., 0] = scaler.transform(xs[..., 0])
        ds = TensorDataset(
            torch.from_numpy(xs),
            torch.from_numpy(scaler.transform(ys_raw).astype(np.float32)),
            torch.from_numpy(ys_raw),
        )
        shuffle = name == "train" and dcfg.get("shuffle_train", False)
        loaders[name] = DataLoader(ds, batch_size=bs, shuffle=shuffle)

    return DataBundle(loaders["train"], loaders["val"], loaders["test"],
                      scaler, x.shape[2], x.shape[3])
