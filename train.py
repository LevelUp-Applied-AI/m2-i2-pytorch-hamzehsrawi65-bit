"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.

    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        self.layer1 = nn.Linear(5, 32)   # 5 input features → 32 hidden units
        self.relu = nn.ReLU()            # activation function
        self.layer2 = nn.Linear(32, 1)   # 32 hidden → 1 output (price prediction)

    def forward(self, x):
        """Define the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 5).

        Returns:
            torch.Tensor: Predictions of shape (N, 1).
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df = pd.read_csv("data/housing.csv")
    print("DataFrame shape:", df.shape)

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]   # keep shape (N, 1)

    # ── 3. Standardize Features ───────────────────────────────────────────────
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    # Why: features have very different scales; standardization ensures
    #      gradient updates are balanced across all input dimensions.

    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    print("X_tensor shape:", X_tensor.shape)
    print("y_tensor shape:", y_tensor.shape)

    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    for epoch in range(num_epochs + 1):  # include epoch 100
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ── 7. Save Predictions ───────────────────────────────────────────────────
    with torch.no_grad():
        preds = model(X_tensor)

    actuals = y_tensor.numpy().flatten()
    predicted = preds.numpy().flatten()

    out_df = pd.DataFrame({'actual': actuals, 'predicted': predicted})
    out_df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()