import numpy as np
import torch
import random

def load_data(data_path, device):
    data = np.load(data_path + "abide.npy", allow_pickle=True).tolist()
    time_series = data["timeseires"]
    feature_matrix = data["corr"]
    label = data["label"]
    time_series = normalize(time_series)
    # we find normalization is important for spectrum features
    time_series = time_series.transpose(0,2,1)
    A = torch.from_numpy(data["pcorr"]).float().to(device) # A shape Size x Time x Node x Node
    label = torch.from_numpy(label).long().to(device)
    time_series = torch.from_numpy(time_series).float().to(device)
    feature_matrix = torch.from_numpy(feature_matrix).float().to(device)
    return time_series, feature_matrix, A, label


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def cal_accuracy(pred, label):
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
        label = torch.from_numpy(label)
    return int(pred.eq(label.reshape(-1)).float().mean().item() * 100)

def cal_specificity_sensitivity(prediction, label):
    tn = np.sum((prediction == 0) & (label == 0))
    fp = np.sum((prediction == 1) & (label == 0))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tp = np.sum((prediction == 1) & (label == 1))
    fn = np.sum((prediction == 0) & (label == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return sensitivity.item(), specificity.item()

def load_index(path):
    return np.load(path + "train_index.npy"), np.load(path + "test_index.npy")


def normalize(x):
    B, N, C = x.shape
    for i in range(B):
        for j in range(N):
            std = np.std(x[i, j]) if np.std(x[i, j]) != 0 else 1e-5
            x[i, j] = (x[i, j] - np.mean(x[i,j])) / std
    return x
