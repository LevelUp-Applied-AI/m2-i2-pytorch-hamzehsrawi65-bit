import json
import time
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Model
# -----------------------------
class HousingModel(nn.Module):
    """Linear(5 -> hidden) -> ReLU -> Linear(hidden -> 1)"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# -----------------------------
# Metrics
# -----------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R^2 = 1 - SS_res / SS_tot
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


# -----------------------------
# Data prep (fixed split for all experiments)
# -----------------------------
def load_and_split(seed: int = 42):
    df = pd.read_csv("data/housing.csv")

    feature_cols = ["area_sqm", "bedrooms", "floor", "age_years", "distance_to_center_km"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[["price_jod"]].values.astype(np.float32)  # (N,1)

    # fixed shuffle/split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    split = int(0.8 * len(X))
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # standardize using TRAIN stats only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


# -----------------------------
# Training (full-batch for speed)
# -----------------------------
def run_experiment(config, X_train, y_train, X_test, y_test):
    hidden_size = config["hidden_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)

    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    model = HousingModel(hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()

    for _ in range(epochs):
        preds = model(Xtr)
        loss = criterion(preds, ytr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_time = time.time() - start

    # final metrics
    with torch.no_grad():
        train_preds = model(Xtr)
        test_preds = model(Xte)

        train_loss = float(criterion(train_preds, ytr).item())
        test_loss = float(criterion(test_preds, yte).item())

        y_true = y_test.reshape(-1)
        y_pred = test_preds.numpy().reshape(-1)

        test_mae = mae(y_true, y_pred)
        test_r2 = r2_score(y_true, y_pred)

    result = {
        "config": config,
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "train_time_sec": float(train_time),
    }
    return result


# -----------------------------
# Plot summary
# -----------------------------
def save_summary_plot(results, out_path="experiment_summary.png"):
    # top 10 by MAE
    top = sorted(results, key=lambda r: r["test_mae"])[:10]

    labels = [
        f"hs={r['config']['hidden_size']}\nlr={r['config']['lr']}\nep={r['config']['epochs']}"
        for r in top
    ]
    maes = [r["test_mae"] for r in top]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(maes)), maes)
    plt.xticks(range(len(maes)), labels, rotation=0)
    plt.ylabel("Test MAE (JOD)")
    plt.title("Top 10 Experiments by Lowest Test MAE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    X_train, y_train, X_test, y_test = load_and_split(seed=SEED)

    # Hyperparameter grid (36 configs >= 30 requirement)
    lr_list = [0.001, 0.01, 0.05]
    hidden_list = [16, 32, 64, 128]
    epoch_list = [50, 100, 150]

    configs = []
    for lr, hs, ep in itertools.product(lr_list, hidden_list, epoch_list):
        configs.append({"lr": lr, "hidden_size": hs, "epochs": ep})

    print(f"Total configs: {len(configs)}")

    results = []
    for i, cfg in enumerate(configs, start=1):
        res = run_experiment(cfg, X_train, y_train, X_test, y_test)
        results.append(res)

        if i % 5 == 0 or i == 1 or i == len(configs):
            print(
                f"[{i:02d}/{len(configs)}] "
                f"MAE={res['test_mae']:.2f} | R2={res['test_r2']:.4f} | "
                f"test_loss={res['final_test_loss']:.2f} | time={res['train_time_sec']:.2f}s "
                f"| cfg={cfg}"
            )

    # Save JSON
    with open("experiments.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Leaderboard Top 10 by lowest MAE
    leaderboard = sorted(results, key=lambda r: r["test_mae"])[:10]
    print("\n=== Leaderboard (Top 10 by lowest Test MAE) ===")
    for rank, r in enumerate(leaderboard, start=1):
        cfg = r["config"]
        print(
            f"{rank:02d}) MAE={r['test_mae']:.2f} | R2={r['test_r2']:.4f} | "
            f"test_loss={r['final_test_loss']:.2f} | time={r['train_time_sec']:.2f}s | "
            f"lr={cfg['lr']} hs={cfg['hidden_size']} ep={cfg['epochs']}"
        )

    # Save plot
    save_summary_plot(results, out_path="experiment_summary.png")
    print("\nSaved experiments.json")
    print("Saved experiment_summary.png")


if __name__ == "__main__":
    main()